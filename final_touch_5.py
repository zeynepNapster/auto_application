import os

# Import necessary modules
import os

def replace_strings_in_file(file_path_in,file_path_ex, replacements):
    """
    Reads a file, replaces specified strings, and saves it under the same name.

    :param file_path: Path to the text file.
    :param replacements: Dictionary where keys are strings to be replaced and values are the replacements.
    """
    try:
        # Read the content of the file
        with open(file_path_in, 'r', encoding='utf-8') as file:
            content = file.read()

        # Replace the strings
        for old_string, new_string in replacements.items():
            content = content.replace(old_string, new_string)

        # Write the modified content back to the same file
        with open(file_path_ex, 'w', encoding='utf-8') as file:
            file.write(content)

        print(f"Replacements completed and saved to {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_path = 'example.txt'
replacements = {
    '(m/w/d)': '',
    '[Your Name]': '',
    'zeynep.tozge@gmail.com': 'tozgezeynep@gmail.com',
    'm/f/d':'',
    '(m/w/x)':'',
    '(f/m/d)':'',
    'm/w/d':'',
}
for file in  os.listdir('./applications'):
    replace_strings_in_file(os.path.join('./applications',file),
                            os.path.join('./final_docs',file),
                            replacements)