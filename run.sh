STAGE_FILES=(
    "stage1.py"
    "stage2.py" 
    "stage3-imbalance.py"
)
VALID_STAGES=(1 2 3)
STAGE=${1:-2}  # Default to stage 2 if no argument provided

if [[ ! " ${VALID_STAGES[@]} " =~ " ${STAGE} " ]]; then
    echo "Invalid stage number. Please choose from: ${VALID_STAGES[@]}"
    exit 1
fi

STAGE_FILE=${STAGE_FILES[$((STAGE-1))]}
python3 src/${STAGE_FILE}
