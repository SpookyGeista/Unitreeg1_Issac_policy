#!/bin/bash

# G1 Policy Deployment Setup mit Miniconda
# Installiert Miniconda und alle notwendigen Abhängigkeiten

echo "=== G1 Policy Deployment Setup mit Miniconda ==="
echo ""

# Prüfe ob Miniconda bereits installiert ist
if command -v conda &> /dev/null; then
    echo "✅ Miniconda ist bereits installiert"
    conda --version
else
    echo "📦 Installiere Miniconda..."
    
    # Download Miniconda für ARM64 (G1)
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
    MINICONDA_SCRIPT="miniconda_installer.sh"
    
    # Lade Miniconda herunter
    wget $MINICONDA_URL -O $MINICONDA_SCRIPT
    
    # Installiere Miniconda
    bash $MINICONDA_SCRIPT -b -p $HOME/miniconda3
    
    # Initialisiere conda
    $HOME/miniconda3/bin/conda init bash
    
    # Lade conda in aktueller Shell
    source $HOME/miniconda3/etc/profile.d/conda.sh
    
    echo "✅ Miniconda installiert"
fi

# Lade conda in aktueller Shell
source $HOME/miniconda3/etc/profile.d/conda.sh

# Erstelle neue conda-Umgebung für Policy-Deployment
echo "🐍 Erstelle conda-Umgebung 'g1_policy'..."
conda create -n g1_policy python=3.8 -y

# Aktiviere Umgebung
conda activate g1_policy

# Installiere PyTorch (CPU Version für G1)
echo "🔥 Installiere PyTorch..."
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Installiere andere Abhängigkeiten
echo "📦 Installiere weitere Abhängigkeiten..."
conda install numpy scipy matplotlib -y
pip install unitree-legged-sdk

# Erstelle Policies-Verzeichnis
echo "📁 Erstelle Policies-Verzeichnis..."
mkdir -p /home/unitree/policies
mkdir -p /home/unitree/logs

# Mache Skripte ausführbar
echo "🔧 Mache Skripte ausführbar..."
chmod +x analyze_policy.py

# Erstelle Aktivierungsskript
echo "📝 Erstelle Aktivierungsskript..."
cat > activate_g1_policy.sh << 'EOF'
#!/bin/bash
# Aktiviert die G1 Policy-Umgebung
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate g1_policy
echo "✅ G1 Policy-Umgebung aktiviert"
echo "Verfügbare Befehle:"
echo "  python analyze_policy.py"
echo "  python g1_policy_deployer.py"
EOF

chmod +x activate_g1_policy.sh

# Teste Installation
echo "🧪 Teste Installation..."
conda activate g1_policy
python -c "import torch; print(f'PyTorch Version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy Version: {numpy.__version__}')"

echo ""
echo "✅ Setup abgeschlossen!"
echo ""
echo "Verwendung:"
echo "1. Aktiviere Umgebung: source activate_g1_policy.sh"
echo "2. Analysiere Policy: python analyze_policy.py"
echo "3. Deploye Policy: python g1_policy_deployer.py"
echo ""
echo "⚠️  Wichtige Hinweise:"
echo "- Verwende immer die conda-Umgebung für Policy-Deployment"
echo "- Teste die Policy immer zuerst ohne Hardware"
echo "- Beginne mit kurzen Testläufen" 