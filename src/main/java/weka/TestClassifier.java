package weka;

import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

public class TestClassifier {

	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource("data/iris.csv.arff");
		Instances data = source.getDataSet();
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);

		Instances train = data;
		
		
		DataSource source2 = new DataSource("data/irisTest.csv.arff");
		Instances data2 = source2.getDataSet();
		if (data2.classIndex() == -1)
			data2.setClassIndex(data.numAttributes() - 1);
		Instances test = data2;


		J48 j48 = new J48();
		j48.setUnpruned(true);

		FilteredClassifier fc = new FilteredClassifier();
		fc.setClassifier(j48);

		// train and make predictions
		fc.buildClassifier(train);
		
		System.out.println("=== my outputs ===");
		for (int i = 0; i < test.numInstances(); i++) {
			double pred = fc.classifyInstance(test.instance(i));
			System.out.print("ID: " + test.instance(i).value(0));
			System.out.print(", actual: " + test.classAttribute().value((int) test.instance(i).classValue()));
			System.out.println(", predicted: " + test.classAttribute().value((int) pred));
		}

		
		//System.out.println(data.toString());
		System.out.println("=== end ===");

	}

}
