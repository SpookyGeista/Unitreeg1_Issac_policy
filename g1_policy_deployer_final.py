#!/usr/bin/env python3
"""
G1 Policy Deployer mit Unitree SDK2
Deployt Isaac Sim trainierte RSL-RL Policies auf den G1
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path

# Unitree SDK2 Imports
try:
    import unitree_sdk2py
    from unitree_sdk2py import idl, utils, core, rpc, go2
    print("✅ Unitree SDK2 geladen")
except ImportError as e:
    print(f"❌ Unitree SDK2 Fehler: {e}")
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

class G1PolicyDeployer:
    def __init__(self, policy_path, config_path=None):
        """
        Initialisiert den G1 Policy Deployer
        
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
        
        # G1 SDK2
        self.g1_client = None
        
        # State Tracking
        self.last_action = np.zeros(12)  # 12 DOF für G1
        self.action_filter = 0.8  # Smoothing Filter
        
        # RSL-RL spezifische Parameter
        self.input_dim = 123
        self.output_dim = 37
        
        print("🤖 G1 Policy Deployer initialisiert")
        
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
                
        except Exception as e:
            print(f"❌ Fehler beim Laden der Policy: {e}")
            raise
            
    def setup_g1_communication(self):
        """Initialisiert die Kommunikation mit dem G1"""
        try:
            print("🔌 Initialisiere G1 Kommunikation...")
            
            # Erstelle G1 Client
            # Hier müssen wir das richtige Interface verwenden
            # Für den Moment verwenden wir einen Mock-Client
            self.g1_client = MockG1Client()
            print("✅ G1 Kommunikation initialisiert")
            
        except Exception as e:
            print(f"❌ Fehler bei G1 Kommunikation: {e}")
            raise
            
    def get_observation(self):
        """
        Sammelt Beobachtungen vom G1 und konvertiert sie für RSL-RL
        Returns: Normalisierte Beobachtungen für die Policy
        """
        # Mock-Beobachtungen für Test
        # In der echten Implementierung würden wir echte G1-Sensordaten verwenden
        obs = np.random.randn(self.input_dim).astype(np.float32)
        
        # Einfache Normalisierung
        obs[4:7] = obs[4:7] / 10.0  # Gyroscope
        obs[7:10] = obs[7:10] / 20.0  # Accelerometer
        obs[10:34] = obs[10:34] / np.pi  # Positionen
        obs[34:58] = obs[34:58] / 20.0   # Geschwindigkeiten
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
        # Mock-Aktion für Test
        print(f"🎯 Wende Aktion an: {action[:3]}... (erste 3 Werte)")
        
        # In der echten Implementierung würden wir das G1 SDK verwenden
        # self.g1_client.send_motor_commands(action)
        
    def run_policy(self, duration=30.0):
        """
        Führt die Policy für eine bestimmte Zeit aus
        Args:
            duration: Laufzeit in Sekunden
        """
        print(f"🚀 Starte RSL-RL Policy-Ausführung für {duration} Sekunden...")
        print("⚠️  Drücke Ctrl+C zum Stoppen")
        print("📝 HINWEIS: Dies ist ein Test-Modus ohne echte G1-Kontrolle")
        
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
        print("✅ Roboter gestoppt (Mock-Modus)")

class MockG1Client:
    """Mock G1 Client für Tests"""
    def __init__(self):
        print("🔧 Mock G1 Client erstellt")
        
    def send_motor_commands(self, action):
        """Mock-Methode für Motor-Befehle"""
        pass

def main():
    """Hauptfunktion"""
    if len(sys.argv) < 2:
        print("Verwendung: python g1_policy_deployer_final.py <policy_path> [duration]")
        print("Beispiel: python g1_policy_deployer_final.py /home/unitree/isaac_policy/model.pt 10")
        sys.exit(1)
        
    policy_path = sys.argv[1]
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    
    # Erstelle Deployer
    deployer = G1PolicyDeployer(policy_path)
    
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