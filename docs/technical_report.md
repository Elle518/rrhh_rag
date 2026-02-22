# RR.HH. RAG - Memoria técnica

## 1. Extracción y limpieza de documentos con Docling

Para la extracción y limpieza de los documentos, utilizamos la herramienta [Docling](https://docling.readthedocs.io/en/latest/). Docling es una biblioteca de Python que facilita la extracción de texto y metadatos de documentos PDF, Word, Excel y otros formatos. Además, ofrece opciones para limpiar el texto extraído, como eliminar imágenes, elementos de diseño, encabezados y pies de página.

Para extraer y limpiar los documentos en PDF, ejecutamos el siguiente comando:

```bash
> python -m scripts.docling_extract_clean \
    --input-dir data/raw \
    --output-dir data/interim/docling/raw \
    --export-markdown
```

Este comando realiza las siguientes acciones:

- Lee los documentos PDF desde el directorio `data/raw`.
- Extrae el texto y los metadatos utilizando Docling.
- Limpia el texto eliminando imágenes, elementos de diseño, encabezados y pies de página.
- Guarda el texto limpio en formato JSON y Markdown en el directorio `data/interim/docling/raw`.

La librería Docling funciona bastante bien con documentos estructurados, como informes, manuales y artículos académicos, preservando la organización natural de los documentos. Sin embargo, es importante tener en cuenta que la calidad de la extracción puede variar dependiendo del formato y la complejidad del documento original.

Así, en nuestro caso, para obtener mejores resulados a la hora de chunkear los documentos, lo ideal es asegurarnos de que los documentos estén bien estructurados, con secciones, subsecciones y párrafos claramente definidos. Esto permitirá que el `HierarchicalChunker` preserve la organización natural del documento durante el proceso de chunking y genere chuncks más pequeños y coherentes.

Esto, en ultimo término, se traduce en una mejor calidad de los embeddings generados y, por ende, en un mejor rendimiento del sistema RAG.

## 2. Revisión y limpieza manual de los documentos

Una vez que los documentos han sido extraídos y limpiados utilizando Docling, es recomendable realizar una revisión manual de los documentos para asegurarnos de que el texto extraído sea de alta calidad y esté libre de errores. Esto es especialmente importante si los documentos originales contienen tablas, gráficos o elementos de diseño complejos que podrían no haber sido manejados correctamente por Docling.

Para ello, nos crearemos un directorio `data/interim/docling/cleaned` donde guardaremos los documentos en Markdown que hayan sido revisados y limpiados manualmente. Este proceso puede incluir:

- Corregir errores de extracción, como texto faltante o mal formateado.
- Eliminar cualquier contenido irrelevante que no haya sido eliminado por Docling.
- Asegurarnos de que la estructura del documento esté bien definida, con secciones, subsecciones y párrafos claramente delimitados. Esto es crucial para que el `HierarchicalChunker` pueda preservar la organización natural del documento durante el proceso de chunking.

Una vez que los documentos hayan sido revisados y limpiados manualmente, generaremos de nuevo los JSONs con la estructura de los `DoclingDocument` a partir de los Markdown limpios utilizando el siguiente comando:

```bash
> python -m scripts.docling_markdown_to_json \
    --input-dir data/interim/docling/cleaned_md \
    --output-dir data/interim/docling/cleaned_json
```

Estos JSONs limpios serán los que utilizaremos para el proceso de chunking con Docling.

## 3. Chunking de los documentos con Docling
