# Oxford-IIIT Pet Dataset Structure

The Oxford-IIIT Pet Dataset contains images, annotations, and metadata for 37 categories of pets (cats and dogs). Below is a description of the directory structure and the contents of each part of the dataset.

## Directory Structure

```
dataset/oxford-iiit-pet/
├── annotations/
│   ├── README
│   ├── list.txt
│   ├── test.txt
│   ├── trainval.txt
│   ├── trimaps/
│   └── xmls/
├── images/
└── .DS_Store
```

## Contents

### images/
- **Description:** Contains all pet images in JPEG format. Each file is named as `<Breed>_<ID>.jpg` (e.g., `Abyssinian_100.jpg`).

### annotations/
- **README:** Documentation about the dataset and annotation formats.
- **list.txt:** List of all images and their class/label information.
- **trainval.txt:** List of images used for training/validation splits.
- **test.txt:** List of images used for test splits.
- **trimaps/**: Pixel-level segmentation masks for each image (PNG format). Each pixel is labeled as:
  - 1: Foreground (pet)
  - 2: Background
  - 3: Not classified
- **xmls/**: Head bounding box annotations for each image in PASCAL VOC XML format.

## Annotation File Formats

### list.txt, trainval.txt, test.txt
Each line (after comments) has the format:
```
<Image> <CLASS-ID> <SPECIES> <BREED-ID>
```
- `Image`: Image file name without extension
- `CLASS-ID`: 1-37 (pet class)
- `SPECIES`: 1=Cat, 2=Dog
- `BREED-ID`: 1-25 for cats, 1-12 for dogs

**Example:**
```
Abyssinian_100 1 1 1
american_bulldog_100 2 2 1
```

### xmls/ (Head Bounding Box Annotations)
Each XML file contains the bounding box for the pet's head in PASCAL VOC format. Example:
```xml
<annotation>
  <folder>OXIIIT</folder>
  <filename>wheaten_terrier_170.jpg</filename>
  <source>
    <database>OXFORD-IIIT Pet Dataset</database>
    <annotation>OXIIIT</annotation>
    <image>flickr</image>
  </source>
  <size>
    <width>500</width>
    <height>330</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>dog</name>
    <pose>Frontal</pose>
    <truncated>0</truncated>
    <occluded>0</occluded>
    <bndbox>
      <xmin>181</xmin>
      <ymin>126</ymin>
      <xmax>299</xmax>
      <ymax>239</ymax>
    </bndbox>
    <difficult>0</difficult>
  </object>
</annotation>
```

### trimaps/ (Segmentation Masks)
- PNG images with the same base name as the corresponding image.
- Pixel values:
  - 1: Foreground (pet)
  - 2: Background
  - 3: Not classified

## References
- O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, "Cats and Dogs," IEEE Conference on Computer Vision and Pattern Recognition, 2012
- For more details, see the `annotations/README` file. 