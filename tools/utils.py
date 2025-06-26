import xml.etree.ElementTree as ET
import re

def fix_xml_string(xml_string: str) -> str:
    # Fix closing tags
    xml_string = xml_string.replace('<\\step>', '</step>')
    xml_string = xml_string.replace('<\\plan>', '</plan>')
    # Optionally fix attribute quotes (for robustness)
    xml_string = re.sub(r'<step id=(\d+)>', r'<step id="\1">', xml_string)
    return xml_string

def parse_plan_xml(xml_string: str) -> dict:
    xml_string = fix_xml_string(xml_string)
    xml_string = f"<root>{xml_string}</root>"
    root = ET.fromstring(xml_string)
    result = {}
    for child in root:
        if child.tag == 'plan':
            steps = []
            for step in child.findall('step'):
                steps.append({'step': step.text.strip() if step.text else ''})
            result['plan'] = steps
        else:
            result[child.tag] = child.text.strip() if child.text else ''
    return result
