#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sys/stat.h>
#include <dirent.h>

namespace fs = std::filesystem;

// Constants
#define MAX_SEQ_LENGTH 1000000 // Reduced to a more manageable size
#define BLOCK_SIZE 16

// Error checking macro
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

__device__ int max1(int a, int b, int c, int d) {
    return max(max(a, b), max(c, d));
}

// CUDA kernel for Smith-Waterman algorithm
__global__ void smithWatermanKernel(char* seq1, char* seq2, int* scoreMatrix, int seq1Len, int seq2Len, int match, int mismatch, int gap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / seq2Len + 1;
    int j = idx % seq2Len + 1;
    
    if (i <= seq1Len && j <= seq2Len) {
        int matrixIdx = i * (seq2Len + 1) + j;
        int diagScore = (i > 1 && j > 1) ? scoreMatrix[matrixIdx - seq2Len - 2] + (seq1[i - 1] == seq2[j - 1] ? match : mismatch) : 0;
        int upScore = (i > 1) ? scoreMatrix[matrixIdx - seq2Len - 1] + gap : 0;
        int leftScore = (j > 1) ? scoreMatrix[matrixIdx - 1] + gap : 0;
        scoreMatrix[matrixIdx] = max1(0, diagScore, upScore, leftScore);
    }
}

// Function to read sequence from FASTA file
char* readSequence(const char* filename, int* length) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    std::string sequence;
    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == '>') continue;  // Skip header line
        sequence += line;
    }

    *length = sequence.length();
    if (*length > MAX_SEQ_LENGTH) {
        std::cerr << "Sequence too long" << std::endl;
        return nullptr;
    }

    char* seq = new char[*length + 1];
    std::strcpy(seq, sequence.c_str());
    return seq;
}

void smithWatermanAlignment(const fs::path& refFile, const fs::path& queryFile, const fs::path& outputFile) {
    std::cout << "Starting Smith-Waterman alignment" << std::endl;
    int refLen, queryLen;
    char *refSeq = readSequence(refFile.string().c_str(), &refLen);
    char *querySeq = readSequence(queryFile.string().c_str(), &queryLen);

    if (refSeq == nullptr || querySeq == nullptr) {
        std::cerr << "Error reading sequences" << std::endl;
        delete[] refSeq;
        delete[] querySeq;
        return;
    }

    std::cout << "Reference sequence length: " << refLen << std::endl;
    std::cout << "Query sequence length: " << queryLen << std::endl;

    // Determine tile size based on available GPU memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t max_elements = free_mem / sizeof(int) / 4; // Use quarter of available memory
    size_t tile_width = std::min(static_cast<size_t>(sqrt(max_elements)), static_cast<size_t>(512)); // Max tile width of 512

    std::cout << "Using tile size: " << tile_width << "x" << tile_width << std::endl;

    char *d_refSeq, *d_querySeq;
    int *d_scoreMatrix;

    // Allocate CUDA memory with error checking
    cudaError_t err = cudaMalloc((void**)&d_refSeq, tile_width * sizeof(char));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for d_refSeq: " << cudaGetErrorString(err) << std::endl;
        delete[] refSeq;
        delete[] querySeq;
        return;
    }

    err = cudaMalloc((void**)&d_querySeq, tile_width * sizeof(char));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for d_querySeq: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_refSeq);
        delete[] refSeq;
        delete[] querySeq;
        return;
    }

    err = cudaMalloc((void**)&d_scoreMatrix, tile_width * tile_width * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for d_scoreMatrix: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_refSeq);
        cudaFree(d_querySeq);
        delete[] refSeq;
        delete[] querySeq;
        return;
    }

    // Open output file for writing
    std::ofstream outFile(outputFile);
    if (!outFile) {
        std::cerr << "Error opening output file" << std::endl;
        cudaFree(d_refSeq);
        cudaFree(d_querySeq);
        cudaFree(d_scoreMatrix);
        delete[] refSeq;
        delete[] querySeq;
        return;
    }

    // Process tiles
    for (size_t i = 0; i < refLen; i += tile_width) {
        for (size_t j = 0; j < queryLen; j += tile_width) {
            size_t current_tile_width = std::min(tile_width, std::min(refLen - i, queryLen - j));
            
            // Copy current tile data to device
            cudaMemcpy(d_refSeq, refSeq + i, current_tile_width * sizeof(char), cudaMemcpyHostToDevice);
            cudaCheckError();
            cudaMemcpy(d_querySeq, querySeq + j, current_tile_width * sizeof(char), cudaMemcpyHostToDevice);
            cudaCheckError();

            dim3 blockDim(16, 16);
            dim3 gridDim((current_tile_width + blockDim.x - 1) / blockDim.x, 
                         (current_tile_width + blockDim.y - 1) / blockDim.y);

            smithWatermanKernel<<<gridDim, blockDim>>>(
                d_refSeq, d_querySeq, d_scoreMatrix, 
                current_tile_width, current_tile_width, 2, -1, -1);
            cudaCheckError();

            cudaDeviceSynchronize();
            cudaCheckError();

            // Copy tile results back to host and write to file
            int* h_scoreMatrix = new int[current_tile_width * current_tile_width];
            cudaMemcpy(h_scoreMatrix, d_scoreMatrix, 
                       current_tile_width * current_tile_width * sizeof(int), 
                       cudaMemcpyDeviceToHost);
            cudaCheckError();

            for (size_t x = 0; x < current_tile_width; x++) {
                for (size_t y = 0; y < current_tile_width; y++) {
                    outFile << h_scoreMatrix[x * current_tile_width + y] << " ";
                }
                outFile << std::endl;
            }

            delete[] h_scoreMatrix;
        }
    }

    outFile.close();

    // Free memory
    delete[] refSeq;
    delete[] querySeq;
    cudaFree(d_refSeq);
    cudaFree(d_querySeq);
    cudaFree(d_scoreMatrix);
}

int main() {
    // At the beginning of main() function
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    std::cout << "Maximum grid size: " << props.maxGridSize[0] << " x " << props.maxGridSize[1] << " x " << props.maxGridSize[2] << std::endl;

    // Hardcoded paths
    fs::path refFile = "data/reference2.txt";
    fs::path queryDir = "data/quseq2";
    fs::path outputDir = "output/align_results";

    // Check if the output directory exists, if not, create it
    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
    }

    std::cout << "Aligning the sequences first" << std::endl;

    if (!fs::exists(queryDir) || !fs::is_directory(queryDir)) {
        std::cerr << "Query directory does not exist or is not a directory: " << queryDir << std::endl;
        return 1;
    }

    std::cout << "Query directory: " << queryDir << std::endl;
    std::cout << "Reading directory " << queryDir << std::endl;

    // Read directory entries
    for (const auto& entry : fs::directory_iterator(queryDir)) {
        if (fs::is_regular_file(entry)) {
            fs::path inputPath = entry.path();
            fs::path outputPath = outputDir / (entry.path().stem().string() + "_alignment.txt");

            std::cout << "Aligning reference2 with " << inputPath << std::endl;
            try {
                smithWatermanAlignment(refFile, inputPath, outputPath);
            } catch (const std::exception& e) {
                std::cerr << "Error during alignment: " << e.what() << std::endl;
                continue;  // Skip to the next file
            }
        }
    }

    std::cout << "Alignment done, proceeding with Spectrum analysis" << std::endl;

    return 0;
}