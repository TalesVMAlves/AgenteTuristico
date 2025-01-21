import os
from docling.convert import convert_pdf_to_markdown

# Caminhos das pastas
pdf_root = 'pdf'
markdown_root = 'markdowns'

# Itera sobre as subpastas em 'pdf'
for city in os.listdir(pdf_root):
    city_pdf_path = os.path.join(pdf_root, city)
    city_markdown_path = os.path.join(markdown_root, city)

    # Verifica se a subpasta é um diretório
    if os.path.isdir(city_pdf_path):
        # Cria a pasta correspondente em 'markdowns' se não existir
        os.makedirs(city_markdown_path, exist_ok=True)

        # Itera sobre os arquivos PDF na subpasta
        for pdf_file in os.listdir(city_pdf_path):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(city_pdf_path, pdf_file)
                markdown_file = pdf_file.replace('.pdf', '.md')
                markdown_path = os.path.join(city_markdown_path, markdown_file)

                # Verifica se o arquivo Markdown já existe
                if os.path.exists(markdown_path):
                    print(f"Arquivo '{markdown_path}' já existe. Pulando conversão.")
                    continue

                try:
                    # Converte o PDF para Markdown
                    convert_pdf_to_markdown(pdf_path, markdown_path)
                    print(f"Convertido '{pdf_path}' para '{markdown_path}' com sucesso.")
                except Exception as e:
                    print(f"Erro ao converter '{pdf_path}': {e}")
    