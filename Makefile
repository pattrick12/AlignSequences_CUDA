# Makefile for managing the project

# Compiler and flags
NVCC = nvcc
PYTHON = python3
CXXFLAGS = -std=c++17 -O2 -Xcompiler -Wall
LDFLAGS = -lcuda -lcublas -lcufft

# Directories
SRCDIR = src
BUILDDIR = build
BINDIR = bin

# Source files
SRCS = $(SRCDIR)/main.cu
OBJS = $(patsubst $(SRCDIR)/%.cu, $(BUILDDIR)/%.o, $(SRCS))

# Executable name
EXEC = $(BINDIR)/sequence_analysis

# Ensure directories exist
$(shell mkdir -p $(BUILDDIR))
$(shell mkdir -p $(BINDIR))

# Default target
all: fetch build visualize

# Fetch sequences
fetch:
	$(PYTHON) $(SRCDIR)/fetch_fasta.py

# Compile CUDA code
build: $(EXEC)

$(EXEC): $(OBJS)
	$(NVCC) $(OBJS) -o $@ $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Run the executable
run: $(EXEC)
	./$(EXEC)

# Visualize results
visualize: run
	$(PYTHON) $(SRCDIR)/visualize.py

# Clean build files
clean:
	rm -f $(BUILDDIR)/*.o $(EXEC) output.txt alignment_visualization.png