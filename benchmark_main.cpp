#include <iostream>
#include <string>
#include <cstring>

#include "Benchmark.h"

// ============================================================
//  Gigapixel Pipeline — Benchmark Runner  (Milestone 3)
//
//  Usage:
//    ./benchmark_pipeline                         <- synthetic full sweep
//    ./benchmark_pipeline --quick                 <- synthetic quick sweep
//    ./benchmark_pipeline --image /path/img.tiff  <- real image benchmark
//    ./benchmark_pipeline --image /path/img.tiff --tilesize 512 --overlap 16
// ============================================================

int main(int argc, char* argv[])
{
    bool        quickMode  = false;
    std::string imagePath  = "";
    int         tileSize   = 512;
    int         overlap    = 0;
    int         pipeDepth  = 8;

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--quick") == 0)
            quickMode = true;
        else if (std::strcmp(argv[i], "--image") == 0 && i + 1 < argc)
            imagePath = argv[++i];
        else if (std::strcmp(argv[i], "--tilesize") == 0 && i + 1 < argc)
            tileSize = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--overlap") == 0 && i + 1 < argc)
            overlap = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--depth") == 0 && i + 1 < argc)
            pipeDepth = std::atoi(argv[++i]);
    }

    std::cout << "=======================================================\n";
    std::cout << "  Gigapixel Pipeline - Benchmark Suite  (Milestone 3)  \n";
    std::cout << "=======================================================\n";

    try
    {
        BenchmarkSuite suite;

        if (!imagePath.empty())
        {
            // ── Real image mode ──────────────────────────────
            std::cout << "[Mode] Real image benchmark\n";
            suite.runWithImage(imagePath, tileSize, overlap, pipeDepth);
            suite.exportCSV("/content/benchmark_results.csv");
        }
        else
        {
            // ── Synthetic sweep mode ─────────────────────────
            std::cout << "  Parameters swept:\n";
            std::cout << "    Image sizes    : 1GP, 10GP" << (quickMode ? "" : ", 50GP") << "\n";
            std::cout << "    Operations     : Grayscale, Rotate90CW\n";
            std::cout << "    Modes          : CPU-Only, GPU-Only, Heterogeneous\n";
            std::cout << "    Tile sizes     : 256, 512" << (quickMode ? "" : ", 1024") << "\n";
            std::cout << "    Overlap        : 0, " << (quickMode ? "32 px" : "16, 32, 64 px") << "\n";
            std::cout << "    Pipeline depth : 4, 8" << (quickMode ? "" : ", 16") << "\n";
            std::cout << "=======================================================\n";

            if (quickMode)
                std::cout << "[Mode] Quick synthetic sweep\n";
            else
                std::cout << "[Mode] Full synthetic sweep\n";

            suite.runAll(quickMode);
            suite.printThroughputTable();
            suite.printLatencyTable();
            suite.printSpeedupTable();
            suite.exportCSV("/content/benchmark_results.csv");
        }

        std::cout << "\n[Done] Results saved to /content/benchmark_results.csv\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "\n[FATAL] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
