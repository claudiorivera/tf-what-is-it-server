const tfnode = require("@tensorflow/tfjs-node");
const mobilenet = require("@tensorflow-models/mobilenet");
const cors = require("cors");
const express = require("express");

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json({ limit: "50mb" }));

let mobilenetModel;

const initMobileNetModel = async () => {
  mobilenetModel = await mobilenet.load();
};

initMobileNetModel();

app.use("/tf", async (req, res) => {
  try {
    const data = req.body.image.split(",")[1];
    const image = tfnode.node.decodeImage(new Buffer.from(data, "base64"));
    const predictions = await mobilenetModel.classify(image);
    res.json(predictions);
  } catch (error) {
    res.status(500).json({ message: error.message || "Something went wrong" });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
