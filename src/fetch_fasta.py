from Bio import SeqIO, Entrez
import os

# Set your email for Entrez
Entrez.email = "prakhar@me.com"  # Replace with your actual email

def fetch_entrez_seq(accession, database="nucleotide"):
    """Fetch a sequence from NCBI's Entrez database."""
    try:
        handle = Entrez.efetch(db=database, id=accession, rettype="fasta", retmode="text")
        record = next(SeqIO.parse(handle, "fasta"))
        handle.close()
        return record
    except Exception as e:
        print(f"Error fetching sequence {accession}: {e}")
        return None

def save_sequence(record, filename):
    """Save a sequence record to a FASTA file."""
    try:
        with open(filename, "w") as f:
            SeqIO.write(record, f, "fasta")
    except IOError as e:
        print(f"Error saving sequence to {filename}: {e}")

def main(reference_accession, query_accessions):
    # Create directory for query sequences if it doesn't exist
    query_dir = os.path.join("..", "data", "quseq")
    os.makedirs(query_dir, exist_ok=True)

    # Fetch and save reference sequence
    reference = fetch_entrez_seq(reference_accession)
    if reference:
        save_sequence(reference, os.path.join("..", "data", "reference.fasta"))
        print(f"Saved reference sequence: {reference_accession}")
    else:
        print("Failed to fetch reference sequence.")

    # Fetch and save query sequences
    for i, acc in enumerate(query_accessions):
        query = fetch_entrez_seq(acc)
        if query:
            filename = os.path.join(query_dir, f"query_{i+1}.fasta")
            save_sequence(query, filename)
            print(f"Saved query sequence: {acc}")
        else:
            print(f"Failed to fetch query sequence: {acc}")

if __name__ == "__main__":
    reference_accession = "NC_000913"  # E. coli K-12 complete genome
    query_accessions = ["NC_002695.2", "NC_002127.1", "NC_002128.1"]  # Some E. coli genes
    main(reference_accession, query_accessions)