#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

#define MAX_SEQ_LENGTH 1000000
#define MAX_PATH 1024

// CUDA kernel for Smith-Waterman algorithm
__global__ void smithWatermanKernel(char* seq1, char* seq2, int* scoreMatrix, int seq1Len, int seq2Len, int match, int mismatch, int gap){
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= seq1Len && j <= seq2Len)
    {
        int idx = i * (seq2Len + 1) + j;
        int diagScore = scoreMatrix[idx - seq2Len - 2] + (seq1[i-1] == seq2[j-1] ? match : mismatch);
        int upScore = scoreMatrix[idx - seq2Len - 1] + gap;
        int leftScore = scoreMatrix[idx - 1] + gap;
        scoreMatrix[idx] = max(0, max(diagScore, max(upScore, leftScore)));
    }
}

// CUDA kernel to prepare sequence data for FFT
__global__ void prepareSequenceForFFT(char* seq, cufftComplex* fftInput, int seqLen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seqLen)
    {
        switch(seq[idx])
        {
            case 'A': fftInput[idx].x = 1; fftInput[idx].y = 0; break;
            case 'C': fftInput[idx].x = -1; fftInput[idx].y = 1; break;
            case 'G': fftInput[idx].x = -1; fftInput[idx].y = -1; break;
            case 'T': fftInput[idx].x = 1; fftInput[idx].y = -1; break;
            default: fftInput[idx].x = 0; fftInput[idx].y = 0; break;
        }
    }
}

// Function to read sequence from FASTA file
char* readSequence(const char* filename, int* length){
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }
    char* sequence = (char*)malloc(MAX_SEQ_LENGTH * sizeof(char));
    *length = 0;
    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '>') continue;  // Skip header line
        for (int i = 0; line[i] != '\0' && line[i] != '\n'; i++) {
            if (*length >= MAX_SEQ_LENGTH) {
                printf("Sequence too long\n");
                fclose(file);
                free(sequence);
                return NULL;
            }
            sequence[(*length)++] = line[i];
        }
    }
    fclose(file);
    return sequence;
}

// Function to perform Smith-Waterman alignment
void smithWatermanAlignment(const char* refFile, const char* queryFile, const char* outputFile){
    int refLen, queryLen;
    char *refSeq = readSequence(refFile, &refLen);
    char *querySeq = readSequence(queryFile, &queryLen);
    if (refSeq == NULL || querySeq == NULL) {
        printf("Error reading sequences\n");
        return;
    }

    char *d_refSeq, *d_querySeq;
    int *d_scoreMatrix;
    cudaMalloc((void**)&d_refSeq, refLen * sizeof(char));
    cudaMalloc((void**)&d_querySeq, queryLen * sizeof(char));
    cudaMalloc((void**)&d_scoreMatrix, (refLen + 1) * (queryLen + 1) * sizeof(int));

    cudaMemcpy(d_refSeq, refSeq, refLen * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_querySeq, querySeq, queryLen * sizeof(char), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((refLen + blockDim.x - 1) / blockDim.x, (queryLen + blockDim.y - 1) / blockDim.y);

    smithWatermanKernel<<<gridDim, blockDim>>>(d_refSeq, d_querySeq, d_scoreMatrix, refLen, queryLen, 2, -1, -1);

    int *scoreMatrix = (int*)malloc((refLen + 1) * (queryLen + 1) * sizeof(int));
    cudaMemcpy(scoreMatrix, d_scoreMatrix, (refLen + 1) * (queryLen + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    FILE* outFile = fopen(outputFile, "w");
    if (outFile == NULL) {
        printf("Error opening output file\n");
        return;
    }
    for (int i = 0; i <= refLen; i++) {
        for (int j = 0; j <= queryLen; j++) {
            fprintf(outFile, "%d ", scoreMatrix[i * (queryLen + 1) + j]);
        }
        fprintf(outFile, "\n");
    }
    fclose(outFile);

    free(refSeq);
    free(querySeq);
    free(scoreMatrix);
    cudaFree(d_refSeq);
    cudaFree(d_querySeq);
    cudaFree(d_scoreMatrix);
}

// Function to perform FFT on a sequence and analyze results
void analyzeSequenceSpectrum(const char* filename, const char* outputFile){
    int seqLen;
    char* seq = readSequence(filename, &seqLen);
    if (seq == NULL) {
        printf("Error reading sequence\n");
        return;
    }

    cufftComplex *d_fftInput, *d_fftOutput;
    cudaMalloc((void**)&d_fftInput, seqLen * sizeof(cufftComplex));
    cudaMalloc((void**)&d_fftOutput, seqLen * sizeof(cufftComplex));

    int blockSize = 256;
    int numBlocks = (seqLen + blockSize - 1) / blockSize;
    prepareSequenceForFFT<<<numBlocks, blockSize>>>(seq, d_fftInput, seqLen);

    cufftHandle plan;
    cufftPlan1d(&plan, seqLen, CUFFT_C2C, 1);
    cufftExecC2C(plan, d_fftInput, d_fftOutput, CUFFT_FORWARD);

    cufftComplex *fftOutput = (cufftComplex*)malloc(seqLen * sizeof(cufftComplex));
    cudaMemcpy(fftOutput, d_fftOutput, seqLen * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    FILE* outFile = fopen(outputFile, "w");
    if (outFile == NULL) {
        printf("Error opening FFT output file\n");
        return;
    }

    fprintf(outFile, "Frequency\tMagnitude\tPotential Feature\n");
    for (int i = 1; i < seqLen / 2; i++) {
        float magnitude = sqrt(fftOutput[i].x * fftOutput[i].x + fftOutput[i].y * fftOutput[i].y);
        float frequency = (float)i / seqLen;
        
        fprintf(outFile, "%.6f\t%.6f\t", frequency, magnitude);
        
        if (frequency >= 0.32 && frequency <= 0.34) {
            fprintf(outFile, "Potential coding region (3-base periodicity)");
        } else if (magnitude > 5 * seqLen / 100) {
            fprintf(outFile, "Strong periodic pattern");
        }
        fprintf(outFile, "\n");
    }

    fclose(outFile);

    cufftDestroy(plan);
    free(seq);
    free(fftOutput);
    cudaFree(d_fftInput);
    cudaFree(d_fftOutput);
}

int main(){
    // Hardcoded paths
    const char* refFile = "../data/reference.fasta";
    const char* queryDir = "../data/quseq";
    const char* outputDir = "../";

    
    

    DIR *dir;
    DIR *dir1;
    struct dirent *ent;
    char inputPath[PATH_MAX];
    char outputPath[PATH_MAX];

    cout<<"Aligning the sequences first"<<endl;
    dir = opendir(queryDir);
    if (dir == NULL) {
        printf("Could not open query directory: %s\n", queryDir);
        return 1;
    }

    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_type == DT_REG) { // If it's a regular file
            snprintf(inputPath, sizeof(inputPath), "%s/%s", queryDir, ent->d_name);
            snprintf(outputPath, sizeof(outputPath), "%s/%s_alignment.txt", outputDir, ent->d_name);

            printf("Aligning reference with %s\n", inputPath);
            smithWatermanAlignment(refFile, inputPath, outputPath);
        }
    }
    cout<<"Alignment done, proceeding with Spectrum analysis"<<endl;
    closedir(dir);
    dir1 = opendir(queryDir);  // We'll use the same queryDir for spectrum analysis
    if (dir1 == NULL) {
        printf("Could not open sequence directory: %s\n", queryDir);
        return 1;
    }

    while ((ent = readdir(dir1)) != NULL) {
        if (ent->d_type == DT_REG) { // If it's a regular file
            snprintf(inputPath, sizeof(inputPath), "%s/%s", queryDir, ent->d_name);
            snprintf(outputPath, sizeof(outputPath), "%s/%s_spectrum.txt", outputDir, ent->d_name);

            printf("Analyzing spectrum of %s\n", inputPath);
            analyzeSequenceSpectrum(inputPath, outputPath);
        }
    }
    
 

    closedir(dir1);
    return 0;
}