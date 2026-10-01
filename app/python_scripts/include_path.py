import argparse
import os

def replace_keyword_in_files(path, files, keyword):
	# Check if we are on windows and if so, replace each \ with /
	if os.name == "nt":
		path = path.replace("\\", "/")

	for file_path in files:
		if not os.path.isfile(file_path):
			print(f"File not found: {file_path}")
			continue
		with open(file_path, 'r', encoding='utf-8') as f:
			content = f.read()
		new_content = content.replace(keyword, path)
		with open(file_path, 'w', encoding='utf-8') as f:
			f.write(new_content)
		print(f"Updated: {file_path}")

def main():
	parser = argparse.ArgumentParser(description="Replace a keyword with a path in multiple files.")
	parser.add_argument('--path', required=True, help='Path to insert')
	parser.add_argument('--files', nargs='+', required=True, help='List of files to modify')
	parser.add_argument('--keyword', required=True, help='Keyword to replace')
	args = parser.parse_args()
	replace_keyword_in_files(args.path, args.files, args.keyword)

if __name__ == "__main__":
	main()
