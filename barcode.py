import cv2
from pyzbar.pyzbar import decode
import mysql.connector

# Function to insert barcode data into MySQL database
def insert_into_db(barcode_data):
    try:
        # Establish connection to MySQL
        connection = mysql.connector.connect(
            host='127.0.0.1',  # Change if necessary
            user='root',  # Replace with your MySQL username
            password='highend@009',  # Replace with your MySQL password
            database='barcode'  # Your database name
        )

        cursor = connection.cursor()
        
        # Query to insert data into table (assumed table structure)
        query = "INSERT INTO barcodes (barcode_value) VALUES (%s)"
        cursor.execute(query, (barcode_data,))
        
        # Commit the transaction
        connection.commit()
        print(f"Inserted {barcode_data} into the database.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        # Close the connection
        if connection.is_connected():
            cursor.close()
            connection.close()

# Function to capture and decode barcodes from camera
def read_barcodes():
    cap = cv2.VideoCapture(0)  # Open the camera feed

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Decode barcodes from the frame
        barcodes = decode(frame)
        for barcode in barcodes:
            # Extract barcode data and convert to string
            barcode_data = barcode.data.decode('utf-8')
            print(f"Detected Barcode: {barcode_data}")
            
            # Insert barcode into MySQL database
            insert_into_db(barcode_data)
        
        # Display the camera feed with the frame
        cv2.imshow('Barcode Reader', frame)
        
        # Break loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start barcode reader
if __name__ == "__main__":
    read_barcodes()
