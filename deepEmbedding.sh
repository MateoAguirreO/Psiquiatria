#!/bin/bash
echo "INICIANDO REPRODUCCIÓN DE CODIGOS DE DEEPLEARNING EN FILA INDIA"

# 2. ANSIEDAD
echo "-> Ansiedad: Embeddings Viejos"
cd "/home/ci2dt2-ai/Proyectos/Psiquiatria/Ansiedad/Scripts_Embedding_viejo" && python dl_script2_architectures_fixed.py

echo "-> Ansiedad: Embeddings Nuevos"
cd "/home/ci2dt2-ai/Proyectos/Psiquiatria/Ansiedad/Scripts_Embedding_nuevo" && python dl_script1_feature_reduction.py && python dl_script2_architectures_fixed.py

echo "¡REPRODUCCIÓN COMPLETADA!"
