const express = require("express");
const app = express();
const PORT = 3000;

// Middleware to parse JSON data
app.use(express.json());

// Simple route for testing
app.get("/api/hello", (req, res) => {
  res.json({ message: "Hello from Node.js server!" });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
