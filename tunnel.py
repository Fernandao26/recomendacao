#!/usr/bin/env python3
"""
Script para compartilhar a aplicação Streamlit via túnel público
Acesse de qualquer dispositivo via link gerado automaticamente
"""

import subprocess
import sys
import time
from pathlib import Path

def main():
    print("\n" + "="*70)
    print("🎬 RECOMENDADOR DE FILMES/SÉRIES - MODO COMPARTILHADO")
    print("="*70)
    
    # Instalar pyngrok se não estiver instalado
    try:
        from pyngrok import ngrok
    except ImportError:
        print("\n📦 Instalando pyngrok...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok", "-q"])
        from pyngrok import ngrok
    
    # Parar qualquer ngrok anterior
    try:
        ngrok.kill()
    except:
        pass
    
    print("\n🚀 Iniciando aplicação Streamlit...")
    print("   Aguarde alguns segundos...\n")
    
    # Iniciar Streamlit em background
    streamlit_cmd = [
        "streamlit", "run", "streamlit_app.py",
        "--server.port=8502",
        "--server.headless=true",
        "--logger.level=error"
    ]
    
    streamlit_process = subprocess.Popen(
        streamlit_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).parent)
    )
    
    # Aguardar Streamlit iniciar
    time.sleep(3)
    
    try:
        # Criar túnel público
        print("🌐 Criando link público...")
        public_url = ngrok.connect(8502, "http")
        
        print("\n" + "="*70)
        print("✅ SUCESSO! Sua aplicação está compartilhada!")
        print("="*70)
        print(f"\n📱 Link para compartilhar com outros dispositivos:")
        print(f"\n   {public_url}\n")
        print("💡 Cole este link em qualquer navegador!")
        print("   Funciona em PC, celular, tablet, etc.\n")
        print("="*70)
        print("\n⏳ Pressione CTRL+C para parar a aplicação\n")
        
        # Manter vivo
        ngrok_process = ngrok.get_ngrok_process()
        ngrok_process.proc.wait()
        
    except Exception as e:
        print(f"\n❌ Erro ao criar túnel: {e}")
        print("   Tentando acesso local...\n")
        print("📌 Acesso local (mesma rede):")
        print("   http://localhost:8502")
        print("   http://10.0.10.103:8502\n")
        
        # Manter Streamlit rodando
        streamlit_process.wait()
        
    finally:
        try:
            streamlit_process.terminate()
            ngrok.kill()
        except:
            pass
        print("\n✋ Aplicação encerrada.")

if __name__ == "__main__":
    main()
