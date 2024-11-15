# Organization-Specific Threat Intelligence System (OSTIS)

## Overview

With the increasing complexity and frequency of cyber attacks, organizations recognize the need for a proactive and targeted approach to safeguard their digital assets and operations. Threat landscapes vary widely depending on industry-specific objectives, geographic presence, workforce size, partnerships, revenue, and digital assets. This diversity necessitates tailored threat intelligence sources. Relying solely on high-volume, generalized threat data can lead to "alert fatigue," reducing the efficacy of threat monitoring systems. Organization-specific threat intelligence (CTI) is therefore essential to strengthen cybersecurity defenses.

This repository presents **OSTIS (Organization-Specific Threat Intelligence System)**, a comprehensive framework for generating and managing tailored Cyber Threat Intelligence data for specific organizations.

## Key Features

- **Custom Web Crawler**: Identifies and gathers relevant CTI data from reliable security blogs and trusted sources.
- **Automated Content Extraction**: Uses deep-learning models to filter relevant threat data automatically.
- **Domain-Specific Classification**: Maps CTI data to industry-specific scenarios such as:
  - Education
  - Finance
  - Government
  - Healthcare
  - Industrial Control Systems (ICS)
  - Internet of Things (IoT)
- **Explainable AI (XAI)**: Leverages SHapley Additive exPlanations (SHAP) to provide insights into model predictions, enhancing trust and interpretability.
- **Organization-Specific Threat Intelligence Knowledge Graph (OSTIKG)**: A visual and interactive representation of identified threats, displaying relationships and patterns.
  
## Citation 
If you use the OSTIS dataset, models, or any other materials from this repository in your research or application, please cite our paper:

@article{arikkat2024ostis,
  title={OSTIS: A novel Organization-Specific Threat Intelligence System},
  author={Arikkat, Dincy R and Vinod, P and KA, Rafidha Rehiman and Nicolazzo, Serena and Nocera, Antonino and Timpau, Georgiana and Conti, Mauro},
  journal={Computers \& Security},
  volume={145},
  pages={103990},
  year={2024},
  publisher={Elsevier}
}

## Repository Structure

The repository is organized as follows:

- `dataset/`                         # Contains datasets for training the relevant content identification model, domain-specific classification model, and NER.
- `code/`                            # Source code and scripts for OSTIS components.
  -  Script for training and evaluating BERT model for relevant content identification.
  -  BERT-based model for mapping threat intelligence to specific domains.
  -  Code for extracting relationships among entities (e.g., attack patterns, malware families).
  -  Named Entity Recognition (NER) model for extracting threat intelligence entities (e.g., malware groups, software tools).


## Usage

### Relevance Classification
1. Modify the dataset path and input labels of BERT variants as needed.

### Domain Classification
1. Ensure the dataset path and input labels are set correctly in BERT code.
2. Run the script:
   ```bash
   python3 Binary_BERT.py


## LICENSE 
This project is licensed under the MIT License. See the LICENSE file for more details.
