"""Script to copy figures and reports to a sample outputs directory."""
import shutil
from pathlib import Path


def create_sample_outputs():
    """Create sample outputs directory structure and copy files."""
    # Define paths
    scripts_dir = Path(__file__).parent
    project_dir = scripts_dir.parent
    sample_dir = project_dir / 'Sample outputs'
    
    # Create main sample outputs directory
    sample_dir.mkdir(exist_ok=True)
    
    # Define subdirectories to process
    subdirs = [
        'demographics',
        'political',
        'reforms',
        'spectrum',
        'values',
        'perspectives'
    ]
    
    # Copy files for each subdirectory
    for subdir in subdirs:
        script_subdir = scripts_dir / subdir
        if not script_subdir.exists():
            print(f"Warning: {subdir} directory not found")
            continue
            
        # Create corresponding sample output subdirectory
        sample_subdir = sample_dir / subdir
        sample_subdir.mkdir(exist_ok=True)
        
        # Copy figures
        src_figures = script_subdir / 'figures'
        if src_figures.exists():
            for fig_file in src_figures.glob('**/*'):
                if fig_file.is_file():
                    # Create subdirectories if they exist in source
                    rel_path = fig_file.relative_to(src_figures)
                    dest_path = sample_subdir / rel_path
                    dest_path.parent.mkdir(exist_ok=True)
                    shutil.copy2(fig_file, dest_path)
                    print(f"Copied figure: {rel_path}")
        
        # Copy reports
        src_reports = script_subdir / 'reports'
        if src_reports.exists():
            for report_file in src_reports.glob('**/*'):
                if report_file.is_file():
                    # Create subdirectories if they exist in source
                    rel_path = report_file.relative_to(src_reports)
                    dest_path = sample_subdir / rel_path
                    dest_path.parent.mkdir(exist_ok=True)
                    shutil.copy2(report_file, dest_path)
                    print(f"Copied report: {rel_path}")
    
    print("\nSample outputs directory structure:")
    print_directory_structure(sample_dir)


def print_directory_structure(path: Path, prefix: str = ""):
    """Print the directory structure in a tree-like format."""
    # Print current directory
    print(f"{prefix}└── {path.name}/")
    
    # Sort items: directories first, then files
    items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        if item.is_dir():
            print_directory_structure(item, new_prefix)
        else:
            print(f"{new_prefix}└── {item.name}")


if __name__ == "__main__":
    create_sample_outputs() 