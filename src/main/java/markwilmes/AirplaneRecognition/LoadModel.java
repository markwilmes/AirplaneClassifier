package markwilmes.AirplaneRecognition;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class LoadModel {
    public MultiLayerNetwork load() throws IOException {
        final Logger log = LoggerFactory.getLogger(AirplaneRecognition.class);

        log.info("Initializing and loading model");
        File trainedModel = new File("trained_airplane_model.zip");

        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(trainedModel);

        return network;
    }
}
