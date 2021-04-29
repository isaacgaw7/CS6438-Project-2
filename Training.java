import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;


public class SparkTest {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .master("local")
                .appName("JavaRandomForestRegressorExample")
                .getOrCreate();

        StructType schema = new StructType()
                .add("fixed acidity", "double")
                .add("volatile acidity", "double")
                .add("citric acid", "double")
                .add("residual sugar", "double")
                .add("chlorides", "double")
                .add("free sulfur dioxide", "int")
                .add("total sulfur dioxide", "int")
                .add("density", "double")
                .add("pH", "double")
                .add("sulphates", "double")
                .add("alcohol", "double")
                .add("quality","int");

        Dataset<Row> data = spark.read()
                .format("csv")
                .schema(schema)
                .option("header","true")
                .option("delimiter",";")
                .load(".\\TrainingDataset.csv");
        data.show();

        StringIndexerModel featureIndexer = new StringIndexer()
                .setInputCol("fixed acidity")
                .setInputCol("volatile acidity")
                .setInputCol("citric acid")
                .setInputCol("residual sugar")
                .setInputCol("chlorides")
                .setInputCol("free sulfur dioxide")
                .setInputCol("total sulfur dioxide")
                .setInputCol("density")
                .setInputCol("pH")
                .setInputCol("sulphates")
                .setInputCol("alcohol")
                .setOutputCol("labels")
                .fit(data);

        RandomForestRegressor rf = new RandomForestRegressor()
                .setLabelCol("fixed acidity")
                .setLabelCol("volatile acidity")
                .setLabelCol("citric acid")
                .setLabelCol("residual sugar")
                .setLabelCol("chlorides")
                .setLabelCol("free sulfur dioxide")
                .setLabelCol("total sulfur dioxide")
                .setLabelCol("density")
                .setLabelCol("pH")
                .setLabelCol("sulphates")
                .setLabelCol("alcohol")
                .setFeaturesCol("feature");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"fixed acidity"})
                .setInputCols(new String[]{"volatile acidity"})
                .setInputCols(new String[]{"citric acid"})
                .setInputCols(new String[]{"residual sugar"})
                .setInputCols(new String[]{"chlorides"})
                .setInputCols(new String[]{"free sulfur dioxide"})
                .setInputCols(new String[]{"total sulfur dioxide"})
                .setInputCols(new String[]{"density"})
                .setInputCols(new String[]{"pH"})
                .setInputCols(new String[]{"sulphates"})
                .setInputCols(new String[]{"alcohol"})
                .setOutputCol("feature");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {assembler, rf});

        PipelineModel model = pipeline.fit(data);

        spark.stop();
    }
}
