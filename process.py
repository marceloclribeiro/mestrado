import os
import nibabel as nib
import numpy as np
import cv2
import shutil

# Definir a seed para reprodutibilidade
seed = 7
np.random.seed(seed)

# Caminhos das pastas de entrada
pasta_imagens = 'subtask1/TrainImage'
pasta_mascaras = 'subtask1/TrainMask'

# Caminhos das pastas de saída (organização final)
pasta_saida_imagens = 'subtask2/images'
pasta_saida_mascaras = 'subtask2/masks/all'
pasta_saida_yolo = 'subtask2/labels'

# Criar pastas de saída se não existirem
for split in ['train', 'test', 'val', 'heldout1', 'heldout2']:
    os.makedirs(os.path.join(pasta_saida_imagens, split), exist_ok=True)
    os.makedirs(os.path.join(pasta_saida_mascaras, split), exist_ok=True)
    os.makedirs(os.path.join(pasta_saida_yolo, split), exist_ok=True)

# Função para converter máscaras para o formato YOLO de segmentação
def mask_to_yolo(mask_path, label_path):
    """Converte uma máscara de segmentação semântica para formato YOLO"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    h, w = mask.shape
    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes > 0]  # Ignorar fundo (0)
    
    yolo_annotations = []
    for class_id in unique_classes:
        binary_mask = (mask == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) < 3:  # Polígonos precisam ter pelo menos 3 pontos
                continue
            
            # Normalizar coordenadas (entre 0 e 1)
            normalized_contour = contour.astype(np.float32) / [w, h]
            flattened = normalized_contour.flatten()
            
            yolo_annotations.append(f"{int(class_id)} " + " ".join(map(str, flattened)))
    
    # Salvar arquivo de anotação
    if yolo_annotations:
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_annotations))

# Função para normalizar imagens
def normalizar_imagem(imagem):
    return cv2.normalize(imagem, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Função para processar e salvar imagens e máscaras
def processar_e_salvar(caminho_imagem, caminho_mascara, caminho_saida_imagem, caminho_saida_mascara, split):
    imagem_nii = nib.load(caminho_imagem)
    mascara_nii = nib.load(caminho_mascara)
    
    imagem = imagem_nii.get_fdata()
    mascara = mascara_nii.get_fdata()
    
    for slice_idx in range(imagem.shape[2]):
        slice_imagem = normalizar_imagem(imagem[:, :, slice_idx])
        slice_mascara = mascara[:, :, slice_idx].astype(np.uint8)
        
        if np.all(slice_imagem == 0):  # Se a imagem for totalmente preta, descarta a máscara
            continue
        
        nome_base = os.path.basename(caminho_imagem).replace('.nii.gz', '')[:-5]
        nome_arquivo = f'{nome_base}_{slice_idx}.png'
        
        caminho_saida_slice_imagem = os.path.join(caminho_saida_imagem, nome_arquivo)
        caminho_saida_slice_mascara = os.path.join(caminho_saida_mascara, nome_arquivo)
        
        cv2.imwrite(caminho_saida_slice_imagem, slice_imagem)
        cv2.imwrite(caminho_saida_slice_mascara, slice_mascara)
        
        # Converter máscara para YOLO
        label_path = os.path.join(pasta_saida_yolo, split, nome_arquivo.replace('.png', '.txt'))
        mask_to_yolo(caminho_saida_slice_mascara, label_path)

# Lista de pacientes e divisão de dados
pacientes = [f'train_{i:04d}' for i in range(1, 362)]
np.random.shuffle(pacientes)
train_pacientes = pacientes[:40]
test_pacientes, val_pacientes = pacientes[40:50], pacientes[50:60]
heldout1_pacientes = pacientes[60:80]
heldout2_pacientes = pacientes[80:100]

# Função para processar imagens e máscaras para um conjunto de pacientes
def processar_pacientes(pacientes, split):
    for paciente in pacientes:
        imagem_arquivo = next((f for f in os.listdir(pasta_imagens) if f.endswith('.nii.gz') and paciente in f), None)
        mascara_arquivo = next((f for f in os.listdir(pasta_mascaras) if f.endswith('.nii.gz') and paciente in f), None)
        
        if imagem_arquivo and mascara_arquivo:
            processar_e_salvar(
                os.path.join(pasta_imagens, imagem_arquivo),
                os.path.join(pasta_mascaras, mascara_arquivo),
                os.path.join(pasta_saida_imagens, split),
                os.path.join(pasta_saida_mascaras, split),
                split
            )

# Processar cada conjunto de pacientes
processar_pacientes(train_pacientes, 'train')
processar_pacientes(test_pacientes, 'test')
processar_pacientes(val_pacientes, 'val')
processar_pacientes(heldout1_pacientes, 'heldout1')
processar_pacientes(heldout2_pacientes, 'heldout2')

print("Processamento e divisão dos dados concluídos!")
