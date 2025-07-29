#!/bin/bash
# Aktiviert die G1 Policy-Umgebung
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate g1_policy
echo "✅ G1 Policy-Umgebung aktiviert"
echo "Verfügbare Befehle:"
echo "  python analyze_policy.py"
echo "  python g1_policy_deployer.py"
