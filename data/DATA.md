# Entity & Relationship Dataset Documentation

## Overview

This document describes the structure and content of `data/data.json`, a cybersecurity threat intelligence dataset containing named entity recognition and relationship extraction from security reports.

**File Details:**
- **Format**: JSONL (JSON Lines)
- **Records**: 1,762 entries
- **Size**: ~1.8MB
- **Domain**: Cybersecurity threat intelligence
- **Purpose**: Structured extraction of entities and relationships from threat reports

## File Structure

The file contains one JSON object per line, each representing a single threat intelligence excerpt with its extracted entities and relationships.

### Record Schema

```json
{
  "instruction": "string - Original threat intelligence text",
  "input": null,
  "output": "string - Structured extraction results"
}
```

### Field Descriptions

#### `instruction` (string)
Contains the original threat intelligence text describing cybersecurity incidents, malware analysis, attack campaigns, IOCs (Indicators of Compromise), or technical analysis. This text serves as the source material for entity and relationship extraction.

#### `input` (null)
Always null in this dataset. Reserved field that is not utilized.

#### `output` (string)
Contains structured extraction results in a standardized format with two main sections:

1. **Named Entities**: Listed as `(entity_name, entity_type)` tuples
2. **Relationships**: Listed as `(subject, predicate, object)` tuples

**Format Example:**
```
Named Entities: (Shuckworm, attacker), (USB propagation malware, tool)
Relationships: (Shuckworm, mount, cyber attacks against Ukraine), (Shuckworm, add, USB propagation malware)
```

## Entity Types Taxonomy

The dataset uses a comprehensive taxonomy of entity types relevant to cybersecurity threat intelligence:

### Primary Entity Types

| Entity Type | Description | Examples |
|-------------|-------------|----------|
| `attacker` | Threat actors, APT groups, cybercriminal organizations | Shuckworm, Flea, APT15, Nickel, Gamaredon |
| `malware` | Malware families, malicious software variants | AsyncRAT, Zerobot, AppleJeus, Pterodo, Graphican |
| `tool` | Legitimate tools used maliciously or for attacks | PowerShell, Mimikatz, Telegram, OneDrive, Microsoft Graph API |
| `file` | Specific files, executables, scripts | JMTTrader.exe, foto.safe, bypass.ps1, plugin_3.dll |
| `IP` | IP addresses used in attacks or infrastructure | 192.119.15.36, 89.34.237.118, 176.65.137.5 |
| `Domain` | Domain names associated with malicious activity | jmttrading[.]org, unioncrypto[.]vip, kupaywallet[.]com |
| `URL` | Web addresses for malicious downloads or C&C | hxxp://192.119.15[DOT]36:880/ftp.exe |
| `technique` | MITRE ATT&CK technique identifiers | T1588.004, T1583.001, T1587.001 |
| `threat type` | Categories or classifications of threats | ransomware attack, backdoor, infostealer |
| `SHA256` | File hashes for malware identification | 450ebd66ba67bb46bf18d122823ff07ef4a7b11afe63b6f269aec9236a1790cd |

### Specialized Entity Types

- **Certificate authorities**: Sectigo, Let's Encrypt
- **Infrastructure providers**: NameCheap Inc., CrownCloud US, LLC
- **Geographic locations**: Ukraine, Americas, Atlanta GA
- **Software components**: MSI Installer, VBS script, PowerShell script

## Relationship Types

The dataset captures various types of relationships between entities using natural language predicates:

### Action Relationships
- **use, uses**: Entity employs a tool or technique
- **mount, execute, run**: Entity performs an action
- **download, install**: Software installation/deployment actions
- **spread, distribute**: Malware propagation actions

### Attribution Relationships
- **aka**: Alternative names for the same entity
- **on behalf of**: Agency/sponsorship relationships
- **controlled by**: Infrastructure ownership

### Communication Relationships
- **communicate with, connect to**: Network communications
- **resolved to**: DNS resolution relationships
- **C2 server**: Command and control relationships

### Containment Relationships
- **based on, evolution of**: Software lineage
- **variants of**: Related malware families
- **identical to**: Equivalent entities

### Targeting Relationships
- **target, focus on**: Attack targeting
- **exfiltrate data to**: Data theft destinations

## Data Usage Guide

### Reading the Data

**Python Example:**
```python
import json

def read_entity_relationship_data(file_path):
    """Read and parse the entity-relationship JSON lines file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            records.append(record)
    return records

# Load the data
data = read_entity_relationship_data('data_raw/entity&relationship(1).json')
print(f"Loaded {len(data)} records")
```

### Parsing Output Fields

**Extract Entities and Relationships:**
```python
import re

def parse_output(output_string):
    """Parse the output field to extract entities and relationships."""
    entities = []
    relationships = []
    
    # Split by major sections
    sections = output_string.split('\n')
    
    current_section = None
    for section in sections:
        section = section.strip()
        if section.startswith('Named Entities:'):
            current_section = 'entities'
            content = section.replace('Named Entities:', '').strip()
            if content:
                entities.extend(extract_tuples(content))
        elif section.startswith('Relationships:'):
            current_section = 'relationships'
            content = section.replace('Relationships:', '').strip()
            if content and not content.startswith('No related'):
                relationships.extend(extract_tuples(content))
    
    return entities, relationships

def extract_tuples(content):
    """Extract tuples from parenthetical format."""
    # Match patterns like (entity, type) or (subject, predicate, object)
    pattern = r'\(([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, content)
    return [(match[0].strip(), match[1].strip()) for match in matches]

# Example usage
record = data[0]
entities, relationships = parse_output(record['output'])
```

### Common Processing Patterns

**Entity Extraction by Type:**
```python
def get_entities_by_type(entities, entity_type):
    """Filter entities by type."""
    return [entity for entity, etype in entities if etype == entity_type]

# Get all IP addresses
ips = get_entities_by_type(entities, 'IP')

# Get all malware names  
malware = get_entities_by_type(entities, 'malware')
```

**Relationship Analysis:**
```python
def get_relationships_by_predicate(relationships, predicate):
    """Filter relationships by predicate."""
    return [rel for rel in relationships if len(rel) == 3 and rel[1] == predicate]

# Find all "use" relationships
use_relations = get_relationships_by_predicate(relationships, 'use')
```

### Data Quality Considerations

1. **Inconsistent Formatting**: Some records may have "No related entities and relations" or similar text
2. **Entity Name Variations**: Same entities may be referenced with slight variations
3. **Relationship Completeness**: Not all possible relationships are explicitly captured
4. **Domain-Specific Language**: Uses cybersecurity terminology and jargon
5. **Defanged IOCs**: URLs and domains are often "defanged" (e.g., `hxxp://` instead of `http://`)

### Recommended Preprocessing

1. **Normalize Entity Names**: Convert to lowercase, remove extra whitespace
2. **Standardize IOCs**: Un-defang URLs and domains for analysis
3. **Filter Empty Records**: Skip records with no entities or relationships
4. **Deduplicate**: Remove duplicate entity-relationship pairs
5. **Validate Types**: Ensure entity types match expected taxonomy

## Use Cases

This dataset is suitable for:

- **Named Entity Recognition (NER)** model training
- **Relationship Extraction** model development  
- **Threat Intelligence Analysis** automation
- **Cybersecurity Knowledge Graph** construction
- **IOC Extraction** pipeline development
- **Security Report Understanding** systems

## Example Record Analysis

**Input Text:**
```
"Shuckworm has also been observed using a new PowerShell script in order to spread its custom backdoor malware, Pterodo, via USB."
```

**Extracted Entities:**
- (Shuckworm, attacker)
- (PowerShell script, tool)  
- (Pterodo, malware)

**Extracted Relationships:**
- (Shuckworm, use, PowerShell script)
- (Shuckworm, spread, Pterodo)

This structured extraction enables automated analysis of threat intelligence reports for security operations and research purposes.