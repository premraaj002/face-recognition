import tkinter as tk
from tkinter import messagebox
import subprocess

class FaceRecognitionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Library Face Recognition System")
        self.root.geometry("500x400")
        
        self.label = tk.Label(root, text="Library Face Recognition System", font=("Arial", 16))
        self.label.pack(pady=10)
        
        self.enroll_button = tk.Button(root, text="Enroll User", command=self.open_enroll_window, font=("Arial", 12))
        self.enroll_button.pack(pady=10)
        
        self.recognize_button = tk.Button(root, text="Recognize Face", command=self.recognize_face, font=("Arial", 12))
        self.recognize_button.pack(pady=10)
        
        self.exit_button = tk.Button(root, text="Exit", command=root.quit, font=("Arial", 12))
        self.exit_button.pack(pady=10)
    
    def open_enroll_window(self):
        self.enroll_window = tk.Toplevel(self.root)
        self.enroll_window.title("Enroll User")
        self.enroll_window.geometry("400x250")
        
        self.name_label = tk.Label(self.enroll_window, text="Enter Name:", font=("Arial", 12))
        self.name_label.pack(pady=5)
        
        self.name_entry = tk.Entry(self.enroll_window, font=("Arial", 12))
        self.name_entry.pack(pady=5)
        
        self.capture_button = tk.Button(self.enroll_window, text="Capture", command=self.capture_face, font=("Arial", 12))
        self.capture_button.pack(pady=10)
    
    def capture_face(self):
        name = self.name_entry.get()
        if not name:
            messagebox.showerror("Error", "Please enter a name before capturing!")
            return
        try:
            messagebox.showinfo("Enroll User", f"Capturing face for {name}")
            subprocess.run(["python", "./faceregister.py", name], check=True)
        except FileNotFoundError:
            messagebox.showerror("Error", "faceregister.py not found!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    
    def recognize_face(self):
        try:
            messagebox.showinfo("Recognize Face", "Launching Recognition System")
            subprocess.run(["python", "./facerecognition.py"], check=True)
        except FileNotFoundError:
            messagebox.showerror("Error", "facerecognition.py not found!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionUI(root)
    root.mainloop()