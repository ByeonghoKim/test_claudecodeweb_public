/*
 * memtrace.cpp — Intel Pin memory-trace tool for stream_add.
 *
 * Tracks every load and store executed by the target process.
 * For each memory access it records:
 *   - instruction address (PC)
 *   - access type  (R = load, W = store)
 *   - data address
 *   - access size  (bytes)
 *
 * Additionally prints a per-function summary of total bytes read/written
 * so you can see exactly how much data the AVX-512 add kernel touches.
 *
 * Build (requires Intel Pin 3.x):
 *   export PIN_ROOT=/path/to/pin-3.31-...
 *   cd $PIN_ROOT/source/tools/MyPinTool   # or any tool directory
 *   cp /path/to/memtrace.cpp .
 *   make PIN_ROOT=$PIN_ROOT obj-intel64/memtrace.so
 *
 * Run:
 *   $PIN_ROOT/pin -t obj-intel64/memtrace.so \
 *       [-o memtrace.out] [-summary 1] [-filter_func timed_add] \
 *       -- ./stream_add_cpp
 *
 * Output modes:
 *   -summary  0  (default) : full per-access log → memtrace.out
 *   -summary  1            : per-function byte-count summary only (fast)
 *   -filter_func <name>    : only trace accesses inside this function
 *                            (use "timed_add" to isolate the AVX-512 kernel)
 *
 * Compatible with Pin 3.21 – 3.31 (C++17, Intel64/IA-32).
 */

#include "pin.H"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Command-line knobs
// ---------------------------------------------------------------------------
KNOB<std::string> KnobOutputFile(
    KNOB_MODE_WRITEONCE, "pintool", "o", "memtrace.out",
    "Output file for memory trace (ignored when -summary 1)");

KNOB<BOOL> KnobSummaryOnly(
    KNOB_MODE_WRITEONCE, "pintool", "summary", "0",
    "Print per-function byte summary only (no per-access log)");

KNOB<std::string> KnobFilterFunc(
    KNOB_MODE_WRITEONCE, "pintool", "filter_func", "",
    "Only trace accesses inside this function name (empty = trace all)");

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
static std::ofstream  out_file;
static bool           g_filter      = false;
static bool           g_in_target   = false;   // inside the filtered function?
static std::string    g_filter_name;

struct FuncStats {
    UINT64 reads  = 0;
    UINT64 writes = 0;
    UINT64 read_bytes  = 0;
    UINT64 write_bytes = 0;
};
static std::map<std::string, FuncStats> g_stats;   // function name -> stats
static PIN_LOCK g_lock;

// ---------------------------------------------------------------------------
// Analysis routines — called at runtime for every memory access
// ---------------------------------------------------------------------------
VOID RecordMemRead(VOID* ip, VOID* addr, UINT32 size, const char* func)
{
    if (g_filter && !g_in_target) return;

    PIN_GetLock(&g_lock, 1);
    g_stats[func].reads++;
    g_stats[func].read_bytes += size;
    if (!KnobSummaryOnly.Value()) {
        out_file << "R " << func
                 << " ip=" << ip
                 << " addr=" << addr
                 << " size=" << size << "\n";
    }
    PIN_ReleaseLock(&g_lock);
}

VOID RecordMemWrite(VOID* ip, VOID* addr, UINT32 size, const char* func)
{
    if (g_filter && !g_in_target) return;

    PIN_GetLock(&g_lock, 1);
    g_stats[func].writes++;
    g_stats[func].write_bytes += size;
    if (!KnobSummaryOnly.Value()) {
        out_file << "W " << func
                 << " ip=" << ip
                 << " addr=" << addr
                 << " size=" << size << "\n";
    }
    PIN_ReleaseLock(&g_lock);
}

// Track entry/exit of the filtered function
VOID OnFuncEntry() { g_in_target = true;  }
VOID OnFuncExit()  { g_in_target = false; }

// ---------------------------------------------------------------------------
// Instrumentation callback — called once per instruction at JIT time
// ---------------------------------------------------------------------------
VOID Instruction(INS ins, VOID* /*v*/)
{
    // Resolve containing function name
    RTN rtn = INS_Rtn(ins);
    std::string func = RTN_Valid(rtn) ? PIN_UndecorateSymbolName(RTN_Name(rtn),
                                            UNDECORATION_NAME_ONLY)
                                      : "<unknown>";
    // Heap-allocate so the pointer stays valid after this scope
    std::string* func_heap = new std::string(func);

    UINT32 mem_ops = INS_MemoryOperandCount(ins);
    for (UINT32 op = 0; op < mem_ops; ++op) {
        if (INS_MemoryOperandIsRead(ins, op)) {
            INS_InsertPredicatedCall(
                ins, IPOINT_BEFORE,
                (AFUNPTR)RecordMemRead,
                IARG_INST_PTR,
                IARG_MEMORYOP_EA, op,
                IARG_MEMORYREAD_SIZE,
                IARG_PTR, func_heap->c_str(),
                IARG_END);
        }
        if (INS_MemoryOperandIsWritten(ins, op)) {
            INS_InsertPredicatedCall(
                ins, IPOINT_BEFORE,
                (AFUNPTR)RecordMemWrite,
                IARG_INST_PTR,
                IARG_MEMORYOP_EA, op,
                IARG_MEMORYWRITE_SIZE,
                IARG_PTR, func_heap->c_str(),
                IARG_END);
        }
    }
}

// Instrument function entry/exit for filtering
VOID Routine(RTN rtn, VOID* /*v*/)
{
    if (!g_filter) return;
    std::string name = PIN_UndecorateSymbolName(RTN_Name(rtn), UNDECORATION_NAME_ONLY);
    if (name.find(g_filter_name) != std::string::npos) {
        RTN_Open(rtn);
        RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)OnFuncEntry, IARG_END);
        RTN_InsertCall(rtn, IPOINT_AFTER,  (AFUNPTR)OnFuncExit,  IARG_END);
        RTN_Close(rtn);
    }
}

// ---------------------------------------------------------------------------
// Fini — print summary
// ---------------------------------------------------------------------------
VOID Fini(INT32 /*code*/, VOID* /*v*/)
{
    const std::string sep(72, '=');
    const std::string dash(72, '-');

    auto& out = KnobSummaryOnly.Value() ? std::cout : out_file;

    out << "\n" << sep << "\n";
    out << "  Memory trace summary\n";
    out << sep << "\n";
    out << std::left
        << std::setw(32) << "Function"
        << std::right
        << std::setw(10) << "Reads"
        << std::setw(12) << "Read MB"
        << std::setw(10) << "Writes"
        << std::setw(12) << "Write MB"
        << std::setw(12) << "Total MB"
        << "\n" << dash << "\n";

    double mb = 1024.0 * 1024.0;
    for (auto& [name, s] : g_stats) {
        double total_mb = (s.read_bytes + s.write_bytes) / mb;
        out << std::left  << std::setw(32) << name.substr(0, 31)
            << std::right
            << std::setw(10) << s.reads
            << std::setw(12) << std::fixed << std::setprecision(1) << s.read_bytes / mb
            << std::setw(10) << s.writes
            << std::setw(12) << s.write_bytes / mb
            << std::setw(12) << total_mb
            << "\n";
    }
    out << sep << "\n";

    if (!KnobSummaryOnly.Value()) out_file.close();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    PIN_InitSymbols();
    if (PIN_Init(argc, argv)) {
        std::cerr << KNOB_BASE::StringKnobSummary() << "\n";
        return EXIT_FAILURE;
    }

    g_filter_name = KnobFilterFunc.Value();
    g_filter      = !g_filter_name.empty();
    PIN_InitLock(&g_lock);

    if (!KnobSummaryOnly.Value()) {
        out_file.open(KnobOutputFile.Value());
        if (!out_file) {
            std::cerr << "Cannot open output file: " << KnobOutputFile.Value() << "\n";
            return EXIT_FAILURE;
        }
    }

    INS_AddInstrumentFunction(Instruction, nullptr);
    RTN_AddInstrumentFunction(Routine, nullptr);
    PIN_AddFiniFunction(Fini, nullptr);

    PIN_StartProgram();   // never returns
    return 0;
}
