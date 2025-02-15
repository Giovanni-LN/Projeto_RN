import os
import shutil

# Diretórios principais
base_dirs = {
    "train": r"C:\Users\caiof\Documentos\Projeto_RN\MyFood Dataset\train",
    "val": r"C:\Users\caiof\Documentos\Projeto_RN\MyFood Dataset\val",
    "test": r"C:\Users\caiof\Documentos\Projeto_RN\MyFood Dataset\test"
}


# Função para organizar imagens em subpastas por classe
def organizar_imagens_por_classe(base_dir):
    for filename in os.listdir(base_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Obtém a classe a partir do nome do arquivo (antes do "_")
            class_name = filename.split("_")[0]
            class_dir = os.path.join(base_dir, class_name)

            # Cria a pasta se não existir
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            # Move a imagem para a subpasta correta
            src_path = os.path.join(base_dir, filename)
            dst_path = os.path.join(class_dir, filename)
            shutil.move(src_path, dst_path)


# Organiza imagens nos três conjuntos
for dataset in base_dirs:
    organizar_imagens_por_classe(base_dirs[dataset])

print("Organização concluída! 🚀 Agora suas imagens estão separadas por classe.")
