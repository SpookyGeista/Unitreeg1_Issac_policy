#!/usr/bin/env python3
"""
Policy Analyzer für Isaac Sim trainierte Policies
Analysiert die Struktur und Parameter der Policy
"""

import torch
import numpy as np
import json
from pathlib import Path

def analyze_policy(policy_path):
    """Analysiert die Policy-Datei"""
    print(f"🔍 Analysiere Policy: {policy_path}")
    print("=" * 50)
    
    try:
        # Lade Policy
        checkpoint = torch.load(policy_path, map_location='cpu')
        
        # Analysiere Struktur
        print("📊 Policy-Struktur:")
        if isinstance(checkpoint, dict):
            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor {list(value.shape)} - {value.dtype}")
                elif hasattr(value, 'state_dict'):
                    state_dict = value.state_dict()
                    total_params = sum(p.numel() for p in value.parameters())
                    print(f"  {key}: Model mit {len(state_dict)} Layern, {total_params:,} Parametern")
                else:
                    print(f"  {key}: {type(value)}")
        else:
            print(f"  Policy ist ein {type(checkpoint)}")
            
        # Extrahiere Policy-Netzwerk
        if 'actor' in checkpoint:
            policy = checkpoint['actor']
            print("\n🎭 Actor-Netzwerk gefunden")
        elif 'policy' in checkpoint:
            policy = checkpoint['policy']
            print("\n🎯 Policy-Netzwerk gefunden")
        elif 'model_state_dict' in checkpoint:
            # RSL-RL Format
            print("\n🎯 RSL-RL Format erkannt")
            # Hier müssen wir das Netzwerk aus dem state_dict rekonstruieren
            # Für den Moment verwenden wir das state_dict direkt
            policy = checkpoint['model_state_dict']
            print("⚠️  State_dict Format - Netzwerk muss rekonstruiert werden")
        else:
            policy = checkpoint
            print("\n📦 Direktes Netzwerk")
            
        # Nur eval() aufrufen wenn es ein Modell ist
        if hasattr(policy, 'eval'):
            policy.eval()
        elif isinstance(policy, dict):
            print("  State_dict Format - kein eval() nötig")
        
        # Analysiere Netzwerk-Architektur
        print("\n🏗️  Netzwerk-Architektur:")
        if isinstance(policy, dict):
            # State_dict Format
            print("  State_dict Keys:")
            for key in list(policy.keys())[:10]:  # Zeige erste 10 Keys
                if 'weight' in key:
                    weight_shape = list(policy[key].shape)
                    print(f"    {key}: {weight_shape}")
            if len(policy.keys()) > 10:
                print(f"    ... und {len(policy.keys()) - 10} weitere Keys")
        elif hasattr(policy, 'named_modules'):
            for name, module in policy.named_modules():
                if hasattr(module, 'weight'):
                    print(f"  {name}: {list(module.weight.shape)}")
                
        # Teste Forward Pass nur wenn es ein Modell ist
        if hasattr(policy, '__call__'):
            print("\n🧪 Forward Pass Test:")
            
            # Teste verschiedene Input-Größen
            test_sizes = [43, 235, 235]  # Typische Größen für G1
            
            for obs_size in test_sizes:
                try:
                    test_obs = torch.randn(1, obs_size, dtype=torch.float32)
                    with torch.no_grad():
                        action = policy(test_obs)
                        
                    print(f"  Input {obs_size} → Output {list(action.shape)}")
                    print(f"    Output Range: [{action.min().item():.3f}, {action.max().item():.3f}]")
                    
                except Exception as e:
                    print(f"  Input {obs_size} → Fehler: {e}")
        else:
            print("\n⚠️  Kein ausführbares Modell - State_dict Format erkannt")
            print("   Das Modell muss rekonstruiert werden für Forward Pass Tests")
            # Definiere test_sizes für die Konfiguration
            test_sizes = [43, 235, 235]
                
        # Analysiere Normalisierungsparameter
        print("\n📏 Normalisierungsparameter:")
        
        has_obs_norm = 'obs_mean' in checkpoint and 'obs_std' in checkpoint
        has_action_norm = 'action_mean' in checkpoint and 'action_std' in checkpoint
        
        if has_obs_norm:
            obs_mean = checkpoint['obs_mean']
            obs_std = checkpoint['obs_std']
            print(f"  ✅ Observation Normalisierung: {list(obs_mean.shape)}")
            print(f"    Mean Range: [{obs_mean.min().item():.3f}, {obs_mean.max().item():.3f}]")
            print(f"    Std Range: [{obs_std.min().item():.3f}, {obs_std.max().item():.3f}]")
            
        if has_action_norm:
            action_mean = checkpoint['action_mean']
            action_std = checkpoint['action_std']
            print(f"  ✅ Action Normalisierung: {list(action_mean.shape)}")
            print(f"    Mean Range: [{action_mean.min().item():.3f}, {action_mean.max().item():.3f}]")
            print(f"    Std Range: [{action_std.min().item():.3f}, {action_std.max().item():.3f}]")
            
        if not has_obs_norm and not has_action_norm:
            print("  ⚠️  Keine Normalisierungsparameter gefunden")
            
        # Erstelle Konfiguration
        config = {
            "policy_path": str(policy_path),
            "has_obs_normalization": has_obs_norm,
            "has_action_normalization": has_action_norm,
            "tested_input_sizes": test_sizes,
            "control_frequency": 500,
            "action_filter": 0.8,
            "safety_limits": {
                "max_joint_velocity": 20.0,
                "max_joint_torque": 33.5,
                "max_joint_position": 1.0
            }
        }
        
        config_path = policy_path.parent / "policy_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"\n📝 Konfiguration gespeichert: {config_path}")
        
        return True, config
        
    except Exception as e:
        print(f"❌ Fehler bei der Analyse: {e}")
        return False, None

def main():
    """Hauptfunktion"""
    policy_path = Path("/home/unitree/isaac_policy/model.pt")
    
    if not policy_path.exists():
        print(f"❌ Policy-Datei nicht gefunden: {policy_path}")
        return
        
    success, config = analyze_policy(policy_path)
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 Policy-Analyse erfolgreich!")
        print("\nNächste Schritte:")
        print("1. Erstelle den G1 Deployer basierend auf der Analyse")
        print("2. Teste die Policy ohne Hardware")
        print("3. Deploye auf dem G1 mit Vorsicht")
    else:
        print("❌ Policy-Analyse fehlgeschlagen")

if __name__ == "__main__":
    main() 