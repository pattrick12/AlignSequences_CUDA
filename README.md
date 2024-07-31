I have used python and C++(CUDA) code to access two genomes and run smith waterman alignment algorithm on it. The sequence was too long so I trimmed it manually. 

The python file used biopython module to install the genomes as txt/fasta files.
Then I used main.cu code to run smith-waterman alignment algorithm to make scoring matrix.
Since the two genomes were of same organism E coli. The scoring matrix is mainly filled up with 0. The scoring matrices are stored in .txt format in the output directory.
Since the data is large, I have used tiling to efficiently process and manage the data
Features

	•	GPU Acceleration: Utilizes CUDA for faster computations.
	•	Batch Processing: Aligns multiple query sequences against a reference sequence.
	•	Tile Processing: Efficient memory usage by processing sequences in manageable tiles.

Requirements

	•	C++ Compiler: Compatible with C++17 or later.
	•	CUDA Toolkit: Version 10.0 or later.
	•	NVIDIA GPU: With CUDA support.

 
