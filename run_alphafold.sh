#!/bin/bash
# Description: AlphaFold non-docker version
# Author: Sanjay Kumar Srikakulam

usage() {
        echo ""
        echo "Please make sure all required parameters are given"
        echo "Usage: $0 <OPTIONS>"
        echo "Required Parameters:"
        echo "-f <fasta_path>       Path to a FASTA file containing sequence. If a FASTA file contains multiple sequences, then it will be folded as a multimer"

        echo "Optional Parameters:"
	echo "-d <data_dir>         Path to directory of supporting data (default: $ALPHAFOLD_DATA_DIR)"
	echo "-o <output_dir>       Path to a directory that will store the results. (default: current work directory)"
        echo "-n <openmm_threads>   OpenMM threads (default: all available cores)"
        echo "-m <model_preset>     Choose preset model configuration - the monomer model, the monomer model with extra ensembling, monomer model with pTM head, or multimer model (default: 'monomer_ptm')"
        echo "-c <db_preset>        Choose preset MSA database configuration - smaller genetic database config (reduced_dbs) or full genetic database config (full_dbs) (default: 'full_dbs')"
        echo "-p <use_precomputed_msas> Whether to read MSAs that have been written to disk. WARNING: This will not check if the sequence, database or configuration have changed (default: 'false')"
        echo "-l <is_prokaryote>    Optional for multimer system, not used by the single chain system. A boolean specifying true where the target complex is from a prokaryote, and false where it is not, or where the origin is unknown. This value determine the pairing method for the MSA (default: 'None')"
        echo "-b <benchmark>        Run multiple JAX model evaluations to obtain a timing that excludes the compilation time, which should be more indicative of the time required for inferencing many proteins (default: 'false')"
	echo "-t <max_template_date> Maximum template release date to consider (ISO-8601 format - i.e. YYYY-MM-DD). Important if folding historical test sets"
        echo ""
        exit 1
}

while getopts ":d:o:f:t:g:n:m:c:p:l:b" i; do
        case "${i}" in
        d)
                data_dir=$OPTARG
        ;;
        o)
                output_dir=$OPTARG
        ;;
        f)
                fasta_path=$OPTARG
        ;;
        t)
                max_template_date=$OPTARG
        ;;
        n)
                openmm_threads=$OPTARG
        ;;
        m)
                model_preset=$OPTARG
        ;;
        c)
                db_preset=$OPTARG
        ;;
        p)
                use_precomputed_msas=$OPTARG
        ;;
        l)
                is_prokaryote=$OPTARG
        ;;
        b)
                benchmark=true
        ;;
        esac
done

# Parse input and set defaults
if [[  "$output_dir" == "" || "$fasta_path" == "" ]] ; then
    usage
fi

if [[  "$output_dir" == "" ]] ; then
    output_dir="."
fi

if [[ "$data_dir" == "" ]] ; then
    data_dir=$ALPHAFOLD_DATA_DIR
fi

if [[ "$max_template_date" == "" ]] ; then
    max_template_date="1901-01-01"
fi

if [[ "$benchmark" == "" ]] ; then
    benchmark=false
fi

if [[ "$model_preset" == "" ]] ; then
    model_preset="monomer_ptm"
fi

if [[ "$model_preset" != "monomer" && "$model_preset" != "monomer_casp14" && "$model_preset" != "monomer_ptm" && "$model_preset" != "multimer" ]] ; then
    echo "Unknown model preset! Using default ('monomer')"
    model_preset="monomer"
fi

if [[ "$db_preset" == "" ]] ; then
    db_preset="full_dbs"
fi

if [[ "$db_preset" != "full_dbs" && "$db_preset" != "reduced_dbs" ]] ; then
    echo "Unknown database preset! Using default ('full_dbs')"
    db_preset="full_dbs"
fi

if [[ "$use_precomputed_msas" == "" ]] ; then
    use_precomputed_msas="false"
fi

# This bash script looks for the run_alphafold.py script in $ALPHAFOLD_DIR, if it does not exist then exits
# If $ALPHAFOLD_DIR is not defined it looks in the current directory
if [[ -z "${ALPHAFOLD_DIR}" ]]; then
    current_working_dir=$(pwd)
    alphafold_script="$current_working_dir/run_alphafold.py"
else
    alphafold_script="$ALPHAFOLD_DIR/run_alphafold.py"
fi

if [ ! -f "$alphafold_script" ]; then
    echo "Alphafold python script $alphafold_script does not exist."
    exit 1
fi

# OpenMM threads control
if [[ "$openmm_threads" ]] ; then
    export OPENMM_CPU_THREADS=$openmm_threads
fi

# TensorFlow control
if [[ -z "${TF_FORCE_UNIFIED_MEMORY}" ]]; then
    export TF_FORCE_UNIFIED_MEMORY='1'
fi

# JAX control
if [[ -z "${XLA_PYTHON_CLIENT_MEM_FRACTION}" ]]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION='4.0'
fi

# Path and user config (change me if required)
uniref90_database_path="$data_dir/uniref90/uniref90.fasta"
uniprot_database_path="$data_dir/uniprot/uniprot.fasta"
mgnify_database_path="$data_dir/mgnify/mgy_clusters_2018_12.fa"
bfd_database_path="$data_dir/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
small_bfd_database_path="$data_dir/small_bfd/bfd-first_non_consensus_sequences.fasta"
uniclust30_database_path="$data_dir/uniclust30/uniclust30_2018_08/uniclust30_2018_08"
pdb70_database_path="$data_dir/pdb70/pdb70"
pdb_seqres_database_path="$data_dir/pdb_seqres/pdb_seqres.txt"
template_mmcif_dir="$data_dir/pdb_mmcif/mmcif_files"
obsolete_pdbs_path="$data_dir/pdb_mmcif/obsolete.dat"

# Binary path (change me if required)
hhblits_binary_path=$(which hhblits)
hhsearch_binary_path=$(which hhsearch)
jackhmmer_binary_path=$(which jackhmmer)
kalign_binary_path=$(which kalign)

command_args="--fasta_paths=$fasta_path --output_dir=$output_dir --max_template_date=$max_template_date --db_preset=$db_preset --model_preset=$model_preset --benchmark=$benchmark --use_precomputed_msas=$use_precomputed_msas --logtostderr"

database_paths="--uniref90_database_path=$uniref90_database_path --mgnify_database_path=$mgnify_database_path --data_dir=$data_dir --template_mmcif_dir=$template_mmcif_dir --obsolete_pdbs_path=$obsolete_pdbs_path"

binary_paths="--hhblits_binary_path=$hhblits_binary_path --hhsearch_binary_path=$hhsearch_binary_path --jackhmmer_binary_path=$jackhmmer_binary_path --kalign_binary_path=$kalign_binary_path"

if [[ $model_preset == "multimer" ]]; then
	database_paths="$database_paths --uniprot_database_path=$uniprot_database_path --pdb_seqres_database_path=$pdb_seqres_database_path"
else
	database_paths="$database_paths --pdb70_database_path=$pdb70_database_path"
fi

if [[ "$db_preset" == "reduced_dbs" ]]; then
	database_paths="$database_paths --small_bfd_database_path=$small_bfd_database_path"
else
	database_paths="$database_paths --uniclust30_database_path=$uniclust30_database_path --bfd_database_path=$bfd_database_path"
fi

if [[ $is_prokaryote ]]; then
	command_args="$command_args --is_prokaryote_list=$is_prokaryote"
fi

# Run AlphaFold with required parameters
$(python $alphafold_script $binary_paths $database_paths $command_args)
