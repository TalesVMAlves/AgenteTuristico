import os
from moviepy.editor import VideoFileClip
import whisper
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from docling.document_converter import DocumentConverter

def transcrever_video_para_pdf(video_path, pdf_path, modelo_whisper):
    """
    Transcreve o áudio de um vídeo e salva a transcrição em formato PDF.
    """
    try:
        # Carregar o vídeo
        video = VideoFileClip(video_path)
        
        # Extrair áudio do vídeo e salvar como um arquivo WAV temporário
        audio_path = "audio_temp.wav"
        video.audio.write_audiofile(audio_path, codec='pcm_s16le')
        
        # Transcrever o áudio
        resultado = modelo_whisper.transcribe(audio_path)
        texto_transcricao = resultado["text"]
        
        # Configurações do PDF
        page_width, page_height = letter
        margin_left = 50
        margin_top = 750
        line_spacing = 14
        line_length = 90  # Limite de caracteres por linha
        
        # Criar o PDF e adicionar o texto transcrito
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(margin_left, margin_top, f"Transcrição do vídeo: {os.path.basename(video_path)}")
        
        y_position = margin_top - 30
        
        for paragrafo in texto_transcricao.split("\n"):
            while len(paragrafo) > 0:
                linha = paragrafo[:line_length]
                paragrafo = paragrafo[line_length:]
                c.drawString(margin_left, y_position, linha.strip())
                y_position -= line_spacing
                
                if y_position < 50:
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y_position = margin_top
        
        c.save()
        os.remove(audio_path)
        print(f"Transcrição concluída e salva em: {pdf_path}")
    except Exception as e:
        print(f"Erro ao transcrever o vídeo '{video_path}': {e}")

def converter_pdf_para_markdown(pdf_path, markdown_path, converter_docling):
    """
    Converte um arquivo PDF para Markdown.
    """
    try:
        resultado = converter_docling.convert(pdf_path)
        conteudo_markdown = resultado.document.export_to_markdown()
        
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(conteudo_markdown)
        
        print(f"Convertido '{pdf_path}' para '{markdown_path}' com sucesso.")
    except Exception as e:
        print(f"Erro ao converter '{pdf_path}': {e}")

def processar_videos_para_pdfs(videos_root_folder, pdf_root_folder, modelo_whisper):
    """
    Processa todos os vídeos nas subpastas de cidades e gera os PDFs correspondentes.
    """
    if not os.path.exists(videos_root_folder):
        print(f"Pasta '{videos_root_folder}' não encontrada. Pulando processamento de vídeos.")
        return
    
    for cidade in os.listdir(videos_root_folder):
        caminho_videos_cidade = os.path.join(videos_root_folder, cidade)
        caminho_pdfs_cidade = os.path.join(pdf_root_folder, cidade)
        
        if os.path.isdir(caminho_videos_cidade):
            os.makedirs(caminho_pdfs_cidade, exist_ok=True)
            
            for arquivo_video in os.listdir(caminho_videos_cidade):
                if arquivo_video.endswith(('.mp4', '.avi', '.mov')):
                    caminho_video = os.path.join(caminho_videos_cidade, arquivo_video)
                    nome_pdf = os.path.splitext(arquivo_video)[0] + '.pdf'
                    caminho_pdf = os.path.join(caminho_pdfs_cidade, nome_pdf)
                    
                    if os.path.exists(caminho_pdf):
                        print(f"PDF '{caminho_pdf}' já existe. Pulando transcrição.")
                        continue
                    
                    transcrever_video_para_pdf(caminho_video, caminho_pdf, modelo_whisper)

def processar_pdfs_para_markdowns(pdf_root_folder, markdown_root_folder, converter_docling):
    """
    Processa todos os PDFs nas subpastas de cidades e gera os arquivos Markdown correspondentes.
    """
    for cidade in os.listdir(pdf_root_folder):
        caminho_pdfs_cidade = os.path.join(pdf_root_folder, cidade)
        caminho_markdowns_cidade = os.path.join(markdown_root_folder, cidade)
        
        if os.path.isdir(caminho_pdfs_cidade):
            os.makedirs(caminho_markdowns_cidade, exist_ok=True)
            
            for arquivo_pdf in os.listdir(caminho_pdfs_cidade):
                if arquivo_pdf.endswith('.pdf'):
                    caminho_pdf = os.path.join(caminho_pdfs_cidade, arquivo_pdf)
                    nome_markdown = os.path.splitext(arquivo_pdf)[0] + '.md'
                    caminho_markdown = os.path.join(caminho_markdowns_cidade, nome_markdown)
                    
                    if os.path.exists(caminho_markdown):
                        print(f"Markdown '{caminho_markdown}' já existe. Pulando conversão.")
                        continue
                    
                    converter_pdf_para_markdown(caminho_pdf, caminho_markdown, converter_docling)

if __name__ == "__main__":
    videos_root_folder = "videos"
    pdf_root_folder = "pdf"
    markdown_root_folder = "markdowns"
    
    # Carregar o modelo Whisper
    modelo_whisper = whisper.load_model("base")
    
    # Inicializar o conversor do Docling
    converter_docling = DocumentConverter()
    
    # Processar vídeos para gerar PDFs
    processar_videos_para_pdfs(videos_root_folder, pdf_root_folder, modelo_whisper)
    
    # Processar PDFs para gerar arquivos Markdown
    processar_pdfs_para_markdowns(pdf_root_folder, markdown_root_folder, converter_docling)
