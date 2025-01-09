import cv2
from pyzbar.pyzbar import decode
import mysql.connector

# Function to connect to the database
def get_db_connection():
    return mysql.connector.connect(
        host='127.0.0.1',      # MySQL server (replace with your host if necessary)
        user='root',  # MySQL username
        password='highend@009',  # MySQL password
        database='barcode'  # Your database name
    )

# Function to check if barcode exists in the database
def barcode_exists(barcode_data):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        query = "SELECT COUNT(*) FROM barcodes WHERE barcode_value = %s"
        cursor.execute(query, (barcode_data,))
        result = cursor.fetchone()

        return result[0] > 0  # Returns True if the barcode exists

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Function to insert barcode data into MySQL database
def insert_into_db(barcode_data):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # Insert the new barcode data into the barcodes table
        query = "INSERT INTO barcodes (barcode_value) VALUES (%s)"
        cursor.execute(query, (barcode_data,))
        
        # Commit the transaction
        connection.commit()
        print(f"Inserted {barcode_data} into the database.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Function to capture and decode barcodes from the camera
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
            
            # Check if barcode exists in the database
            if barcode_exists(barcode_data):
                print(f"Barcode {barcode_data} already exists in the database.")
            else:
                # Insert barcode into MySQL database if it doesn't exist
                insert_into_db(barcode_data)
        
        # Display the camera feed
        cv2.imshow('Barcode Reader', frame)
        
        # Break loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start barcode reader
if __name__ == "__main__":
    read_barcodes()
