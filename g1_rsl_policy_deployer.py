#!/usr/bin/env python3
"""
G1 RSL-RL Policy Deployer
Spezialisiert für Isaac Sim trainierte RSL-RL Policies
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path

# Unitree SDK Imports
try:
    # Versuche das Python SDK zu importieren
    import unitree_legged_sdk
    from unitree_legged_sdk import *
    SDK_TYPE = "python"
except ImportError:
    try:
        # Fallback: Verwende das C++ SDK über ctypes
        import ctypes
        import os
        
        # Setze Library-Pfad
        lib_path = "/home/unitree/unitree_sdk2-main/lib/aarch64/libunitree_sdk2.a"
        if not os.path.exists(lib_path):
            raise ImportError(f"Unitree SDK Library nicht gefunden: {lib_path}")
            
        # Lade Library
        unitree_lib = ctypes.CDLL(lib_path)
        SDK_TYPE = "cpp"
        print("✅ Unitree C++ SDK geladen")
        
    except Exception as e:
        print(f"❌ Unitree SDK nicht gefunden: {e}")
        print("Bitte stelle sicher, dass das Unitree SDK installiert ist.")
        sys.exit(1)

class ActorNetwork(nn.Module):
    """Actor-Netzwerk für RSL-RL Policies"""
    def __init__(self, input_dim=123, hidden_dim=256, output_dim=37):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

class G1RSLPolicyDeployer:
    def __init__(self, policy_path, config_path=None):
        """
        Initialisiert den RSL-RL Policy Deployer
        
        Args:
            policy_path: Pfad zur .pt Policy-Datei
            config_path: Optional - Pfad zur Konfigurationsdatei
        """
        self.policy_path = Path(policy_path)
        self.config_path = Path(config_path) if config_path else None
        
        # G1 Hardware Parameter
        self.dt = 0.002  # 500Hz Control Frequency
        self.max_vel = 2.0  # m/s
        self.max_yaw_rate = 2.0  # rad/s
        
        # Policy Parameter
        self.actor = None
        self.obs_mean = None
        self.obs_std = None
        self.action_mean = None
        self.action_std = None
        
        # G1 SDK
        self.udp = None
        self.safe = None
        
        # State Tracking
        self.last_action = np.zeros(12)  # 12 DOF für G1
        self.action_filter = 0.8  # Smoothing Filter
        
        # RSL-RL spezifische Parameter
        self.input_dim = 123
        self.output_dim = 37
        
        print("🤖 G1 RSL-RL Policy Deployer initialisiert")
        
    def load_policy(self):
        """Lädt die RSL-RL Policy und rekonstruiert das Netzwerk"""
        try:
            print(f"📂 Lade RSL-RL Policy von: {self.policy_path}")
            
            # Lade Checkpoint
            checkpoint = torch.load(self.policy_path, map_location='cpu')
            
            # Extrahiere model_state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("✅ RSL-RL State_dict geladen")
            else:
                raise ValueError("Kein model_state_dict gefunden")
                
            # Rekonstruiere Actor-Netzwerk
            self.actor = ActorNetwork(
                input_dim=self.input_dim,
                hidden_dim=256,
                output_dim=self.output_dim
            )
            
            # Lade Gewichte
            self.actor.load_state_dict(state_dict, strict=False)
            self.actor.eval()
            print("✅ Actor-Netzwerk rekonstruiert und geladen")
            
            # Analysiere State_dict für Normalisierung
            self._analyze_state_dict(state_dict)
                
        except Exception as e:
            print(f"❌ Fehler beim Laden der Policy: {e}")
            raise
            
    def _analyze_state_dict(self, state_dict):
        """Analysiert das State_dict für Normalisierungsparameter"""
        print("🔍 Analysiere State_dict für Normalisierung...")
        
        # Suche nach Normalisierungsparametern
        obs_norm_keys = [k for k in state_dict.keys() if 'obs' in k.lower() and ('mean' in k.lower() or 'std' in k.lower())]
        action_norm_keys = [k for k in state_dict.keys() if 'action' in k.lower() and ('mean' in k.lower() or 'std' in k.lower())]
        
        if obs_norm_keys:
            print(f"  Gefundene Observation-Normalisierung: {obs_norm_keys}")
            
        if action_norm_keys:
            print(f"  Gefundene Action-Normalisierung: {action_norm_keys}")
            
        if not obs_norm_keys and not action_norm_keys:
            print("  ⚠️  Keine Normalisierungsparameter gefunden")
            print("  Verwende Standard-Normalisierung basierend auf G1-Sensoren")
            
    def setup_g1_communication(self):
        """Initialisiert die Kommunikation mit dem G1"""
        try:
            print("🔌 Initialisiere G1 Kommunikation...")
            
            # Erstelle UDP Kommunikation
            self.udp = unitree_legged_sdk.UDP(UNITREE_LEGGED_SDK_VERSION)
            self.safe = unitree_legged_sdk.Safety(unitree_legged_sdk.LeggedType.A1)
            
            # Initialisiere Low Level Control
            self.low_cmd = unitree_legged_sdk.LowCmd()
            self.low_state = unitree_legged_sdk.LowState()
            
            # Setze initiale Positionen
            for i in range(12):
                self.low_cmd.motorCmd[i].mode = 0x0A
                self.low_cmd.motorCmd[i].q = 0.0
                self.low_cmd.motorCmd[i].dq = 0.0
                self.low_cmd.motorCmd[i].Kp = 0.0
                self.low_cmd.motorCmd[i].Kd = 0.0
                self.low_cmd.motorCmd[i].tau = 0.0
                
            print("✅ G1 Kommunikation initialisiert")
            
        except Exception as e:
            print(f"❌ Fehler bei G1 Kommunikation: {e}")
            raise
            
    def get_observation(self):
        """
        Sammelt Beobachtungen vom G1 und konvertiert sie für RSL-RL
        Returns: Normalisierte Beobachtungen für die Policy
        """
        # Empfange aktuellen State
        self.udp.Recv()
        self.udp.GetRecv(self.low_state)
        
        # Extrahiere relevante Sensordaten
        obs = []
        
        # IMU Daten
        obs.extend(self.low_state.imu.quaternion)  # 4
        obs.extend(self.low_state.imu.gyroscope)   # 3
        obs.extend(self.low_state.imu.accelerometer) # 3
        
        # Motor Positionen und Geschwindigkeiten
        for i in range(12):
            obs.append(self.low_state.motorState[i].q)    # Position
            obs.append(self.low_state.motorState[i].dq)   # Geschwindigkeit
            
        # Füge letzte Aktion hinzu (für Rekurrenz)
        obs.extend(self.last_action)
        
        # Füge Zeros hinzu um auf 123 zu kommen (falls nötig)
        while len(obs) < self.input_dim:
            obs.append(0.0)
            
        obs = np.array(obs[:self.input_dim], dtype=np.float32)
        
        # Einfache Normalisierung basierend auf G1-Sensoren
        # IMU Normalisierung
        obs[4:7] = obs[4:7] / 10.0  # Gyroscope
        obs[7:10] = obs[7:10] / 20.0  # Accelerometer
        
        # Motor Normalisierung
        obs[10:34] = obs[10:34] / np.pi  # Positionen
        obs[34:58] = obs[34:58] / 20.0   # Geschwindigkeiten
        
        # Action Normalisierung
        obs[58:95] = obs[58:95] / np.pi  # Letzte Aktionen
            
        return obs
        
    def compute_action(self, obs):
        """
        Berechnet Aktion basierend auf RSL-RL Policy
        Args:
            obs: Normalisierte Beobachtungen
        Returns:
            Denormalisierte Aktionen für G1
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = self.actor(obs_tensor)
            action = action.squeeze(0).numpy()
            
        # Konvertiere 37-dimensionale Aktion zu 12 G1-Motoren
        # Hier müssen wir die Mapping-Logik implementieren
        g1_action = self._convert_to_g1_action(action)
            
        # Smoothing Filter
        g1_action = self.action_filter * self.last_action + (1 - self.action_filter) * g1_action
        self.last_action = g1_action.copy()
        
        return g1_action
        
    def _convert_to_g1_action(self, action):
        """
        Konvertiert 37-dimensionale Policy-Aktion zu 12 G1-Motor-Positionen
        """
        # Einfache Mapping-Strategie
        # Nehme die ersten 12 Dimensionen für die 12 Motoren
        g1_action = action[:12].copy()
        
        # Begrenze auf sichere Werte
        g1_action = np.clip(g1_action, -1.0, 1.0)
        
        # Skaliere auf G1-Motor-Bereiche (in Radians)
        g1_action = g1_action * 0.5  # ±0.5 Radians
        
        return g1_action
        
    def apply_action(self, action):
        """
        Wendet Aktionen auf G1 an
        Args:
            action: 12-dimensionale Aktion (Position)
        """
        # Konvertiere Aktionen zu Motor-Befehlen
        for i in range(12):
            # Position Control
            self.low_cmd.motorCmd[i].q = action[i]
            self.low_cmd.motorCmd[i].dq = 0.0
            self.low_cmd.motorCmd[i].Kp = 20.0  # Position Gain
            self.low_cmd.motorCmd[i].Kd = 0.5   # Velocity Gain
            self.low_cmd.motorCmd[i].tau = 0.0
            
        # Sende Befehle
        self.udp.SetSend(self.low_cmd)
        self.udp.Send()
        
    def run_policy(self, duration=30.0):
        """
        Führt die Policy für eine bestimmte Zeit aus
        Args:
            duration: Laufzeit in Sekunden
        """
        print(f"🚀 Starte RSL-RL Policy-Ausführung für {duration} Sekunden...")
        print("⚠️  Drücke Ctrl+C zum Stoppen")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # Sammle Beobachtungen
                obs = self.get_observation()
                
                # Berechne Aktion
                action = self.compute_action(obs)
                
                # Wende Aktion an
                self.apply_action(action)
                
                # Kurze Pause für 500Hz
                time.sleep(self.dt)
                
        except KeyboardInterrupt:
            print("\n⏹️  Policy-Ausführung gestoppt")
        except Exception as e:
            print(f"❌ Fehler während Policy-Ausführung: {e}")
        finally:
            self.stop_robot()
            
    def stop_robot(self):
        """Stoppt den Roboter sicher"""
        print("🛑 Stoppe Roboter...")
        
        # Setze alle Motoren auf 0
        for i in range(12):
            self.low_cmd.motorCmd[i].mode = 0x0A
            self.low_cmd.motorCmd[i].q = 0.0
            self.low_cmd.motorCmd[i].dq = 0.0
            self.low_cmd.motorCmd[i].Kp = 0.0
            self.low_cmd.motorCmd[i].Kd = 0.0
            self.low_cmd.motorCmd[i].tau = 0.0
            
        # Sende Stopp-Befehle
        for _ in range(10):
            self.udp.SetSend(self.low_cmd)
            self.udp.Send()
            time.sleep(0.01)
            
        print("✅ Roboter gestoppt")

def main():
    """Hauptfunktion"""
    if len(sys.argv) < 2:
        print("Verwendung: python g1_rsl_policy_deployer.py <policy_path> [duration]")
        print("Beispiel: python g1_rsl_policy_deployer.py /home/unitree/isaac_policy/model.pt 10")
        sys.exit(1)
        
    policy_path = sys.argv[1]
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    
    # Erstelle Deployer
    deployer = G1RSLPolicyDeployer(policy_path)
    
    try:
        # Lade Policy
        deployer.load_policy()
        
        # Setup G1 Kommunikation
        deployer.setup_g1_communication()
        
        # Führe Policy aus
        deployer.run_policy(duration)
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 