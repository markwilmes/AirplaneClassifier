package markwilmes.AirplaneRecognition;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultipleEpochsIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

// create a gui or web-page that you can put a sr-71, b2, or b1 picture in, and it will classify that picture.

public class AirplaneRecognition {


    private static final Logger log = LoggerFactory.getLogger(AirplaneRecognition.class);

    protected static int channels = 3;

    private static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private static ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private static SubsamplingLayer maxPool(String name,  int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    public DataSetIterator LoadFiles(String filePath, int seed, int numExamples,int numLabels,int batchSize, double splitTrainTest) throws IOException {
        Random rand = new Random(seed);
        log.info("Load data....");
        //File airplanes = new File("../../airplanes/");
        File airplanes = new File(filePath);
        FileSplit airplane_split = new FileSplit(airplanes, NativeImageLoader.ALLOWED_FORMATS, rand);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        System.out.println(labelMaker.getLabelForPath(filePath));
        BalancedPathFilter pathFilter = new BalancedPathFilter(rand, labelMaker, numExamples, numLabels, batchSize);


        InputSplit[] inputSplit = airplane_split.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        ImageTransform flipTransform1 = new FlipImageTransform(rand);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(10101));
        ImageTransform warpTransform = new WarpImageTransform(rand, 42);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, warpTransform, flipTransform2});

        ImageRecordReader recordReader = new ImageRecordReader(100,100,3,labelMaker);
        recordReader.initialize(trainData);

        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,numLabels);

        return testIter;
    }

    public MultiLayerConfiguration configureNetwork(int seed, int numLabels){
        Configuration cudoConf =  CudaEnvironment.getInstance().getConfiguration()
                .allowMultiGPU(true);
        log.info("Device list " + cudoConf.getAvailableDevices().toString());

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.005)
                .activation(Activation.RELU)
                .updater(new Sgd(0.0001))
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.9))
                .list()
                .layer(0, convInit("cnn1", channels, 50 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2,2}))
                .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxool2", new int[]{2,2}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(100, 100, channels))
                .build();

        return conf;
    }


    public static void main(String[] args) throws Exception {
        ReadConfig read = new ReadConfig();

        AirplaneRecognition airplane = new AirplaneRecognition();

        HashMap map = read.config();
        int numLabels = Integer.parseInt((String) map.get("numLabels"));
        int numExamples = Integer.parseInt((String) map.get("numExamples"));
        int batchSize = Integer.parseInt((String) map.get("batchSize"));
        int seed = Integer.parseInt((String) map.get("seed"));
        double splitTrainTest = Double.parseDouble((String) map.get("splitTrainTest"));
        int epochs = Integer.parseInt((String) map.get("numEpochs"));
        String filePath = (String)map.get("filePath");
        int load = Integer.parseInt((String)map.get("loadModel"));

        DataSetIterator testIter = airplane.LoadFiles(filePath,seed,numExamples,numLabels,batchSize,splitTrainTest);
        DataSet allData = ((RecordReaderDataSetIterator) testIter).next();

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);

        MultipleEpochsIterator trainIter = new MultipleEpochsIterator(epochs, testIter);

    /*
        Create an iterator using the batch size for one iteration
     */
        if(load == 0){
            MultiLayerConfiguration conf = airplane.configureNetwork(seed,numLabels);

            //Initialize the user interface backend
            UIServer uiServer = UIServer.getInstance();

            //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
            StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

            //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
            uiServer.attach(statsStorage);

            //Then add the StatsListener to collect this information from the network, as it trains

            MultiLayerNetwork network = new MultiLayerNetwork(conf);
            network.setListeners(new StatsListener(statsStorage));
            network.fit(trainIter);

            log.info("Saving model");
            File TrainedModel = new File("trained_airplane_model.zip");
            boolean updater = true;

            ModelSerializer.writeModel(network,TrainedModel,updater);

            Evaluation eval = network.evaluate(testIter);
            log.info(eval.stats(true));
        }else{
            log.info("Loading network model");
            LoadModel model = new LoadModel();
            MultiLayerNetwork network = model.load();
            log.info("Identifying aircraft");
            // Use prediction with dataset here network.predict(airplane);
        }
        log.info("FINISHED");
    }
}

