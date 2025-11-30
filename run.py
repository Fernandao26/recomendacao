#!/usr/bin/env python3
"""
Script para compartilhar via acesso local na rede
Use este método se todos os dispositivos estão na mesma rede Wi-Fi
"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*70)
print("🎬 RECOMENDADOR DE FILMES/SÉRIES")
print("="*70)
print("\n📱 COMO ACESSAR EM OUTRO DISPOSITIVO:\n")
print("1️⃣  Certifique-se que está na MESMA WI-FI")
print("2️⃣  Abra o navegador no outro dispositivo")
print("3️⃣  Cole um dos links abaixo:\n")
print("   • http://localhost:8502 (se for o mesmo PC)")
print("   • http://10.0.10.103:8502 (outro PC/celular na rede)\n")
print("="*70)
print("\n🚀 Iniciando aplicação...\n")

# Executar Streamlit
subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    "streamlit_app.py",
    "--server.port=8502",
    "--logger.level=error"
])
