import React, { useState } from "react";
import { View, Text, Button, Image, StyleSheet, ActivityIndicator } from "react-native";
import * as ImagePicker from "expo-image-picker";

export default function HomeScreen() {

  const [image, setImage] = useState<string | null>(null);
  const [stage, setStage] = useState("");
  const [confidence, setConfidence] = useState("");
  const [harvest, setHarvest] = useState("");
  const [loading, setLoading] = useState(false);

  const pickImage = async () => {

    const permission = await ImagePicker.requestCameraPermissionsAsync();

    if (!permission.granted) {
      alert("Camera permission is required.");
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      quality: 0.6,
      allowsEditing: false
    });

    if (result.canceled) return;

    const uri = result.assets[0].uri;
    setImage(uri);

    setLoading(true);

    try {

      const formData = new FormData();

      formData.append("file", {
        uri: uri,
        name: "photo.jpg",
        type: "image/jpeg"
      } as any);

      const response = await fetch("http://10.3.8.6:8000/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (data.detections && data.detections.length > 0) {

        const best = data.detections[0];

        setStage(best.stage_name);
        setConfidence(best.confidence + "%");

      } else {
        setStage("No fruit detected");
        setConfidence("");
      }

      setHarvest(data.harvest_ready ? "Harvest Ready" : "Not Ready");

    } catch (error) {

      alert("Server connection failed.");

    }

    setLoading(false);
  };

  return (

    <View style={styles.container}>

      <Text style={styles.title}>
        Pomegranate Maturity Detector
      </Text>

      <Button title="Take Photo" onPress={pickImage} />

      {image && (
        <Image source={{ uri: image }} style={styles.image} />
      )}

      {loading && <ActivityIndicator size="large" style={{marginTop:20}} />}

      {stage !== "" && !loading && (
        <View style={styles.resultBox}>

          <Text style={styles.resultText}>
            Stage: {stage}
          </Text>

          <Text style={styles.resultText}>
            Confidence: {confidence}
          </Text>

          <Text
            style={[
              styles.harvest,
              harvest === "Harvest Ready" ? styles.ready : styles.notReady
            ]}
          >
            {harvest}
          </Text>

        </View>
      )}

    </View>

  );
}

const styles = StyleSheet.create({

  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
    backgroundColor: "#ffffff"
  },

  title: {
    fontSize: 26,
    fontWeight: "bold",
    marginBottom: 25
  },

  image: {
    width: 300,
    height: 300,
    marginTop: 20,
    borderRadius: 10
  },

  resultBox: {
    marginTop: 25,
    alignItems: "center"
  },

  resultText: {
    fontSize: 18,
    marginTop: 5
  },

  harvest: {
    fontSize: 22,
    fontWeight: "bold",
    marginTop: 15
  },

  ready: {
    color: "green"
  },

  notReady: {
    color: "red"
  }

});