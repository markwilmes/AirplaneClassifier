package markwilmes.AirplaneRecognition;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Date;
import java.util.HashMap;
import java.util.Properties;

public class ReadConfig {

    InputStream inputStream;

    public HashMap config() throws IOException{
        HashMap map = new HashMap();
        try {
            Properties prop = new Properties();
            String propFileName = "config.properties";


            inputStream = getClass().getClassLoader().getResourceAsStream(propFileName);

            if (inputStream != null) {
                prop.load(inputStream);
            } else {
                throw new FileNotFoundException("property file '" + propFileName + "' not found in the classpath");
            }

            Date time = new Date(System.currentTimeMillis());

            // get the property value and print it out
            map.put("channels",prop.getProperty("channels"));
            map.put("numLabels",prop.getProperty("num_labels"));
            map.put("numExamples",prop.getProperty("num_examples"));
            map.put("batchSize",prop.getProperty("batch_size"));
            map.put("numEpochs",prop.getProperty("num_epochs"));
            map.put("seed",prop.getProperty("seed"));
            map.put("splitTrainTest",prop.getProperty("split_train_test"));
            map.put("filePath",prop.getProperty("file_path"));
            map.put("loadModel",prop.getProperty("load_model"));

        } catch (Exception e) {
            System.out.println("Exception: " + e);
        } finally {
            inputStream.close();
        }

        return map;
    }
}
