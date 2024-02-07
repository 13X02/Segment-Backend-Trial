const express = require("express");
const { PythonShell } = require("python-shell");
const multer = require("multer");
const fs = require("fs");

const app = express();
const port = 3001;

// Set up multer for handling file uploads
const upload = multer({ dest: "uploads/" });

// Middleware to parse JSON requests
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Middleware to allow CORS
app.use((req, res, next) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    next();
});

// Route to handle image prediction
app.post("/predict", upload.single("image"), (req, res) => {
    const imageFilePath = req.file.path;

    // Run Python script to perform inference
    PythonShell.run("predict.py", { args: [imageFilePath] }, (err, result) => {
        // Delete the uploaded image file
        fs.unlink(imageFilePath, (err) => {
            if (err) {
                console.error("Error deleting file:", err);
            }
        });

        if (err) {
            console.error(err);
            res.status(500).send("Internal Server Error");
        } else {
            const parsedResult = JSON.parse(result[0]);
            res.json(parsedResult);
        }
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
