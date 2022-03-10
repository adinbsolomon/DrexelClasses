import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class Output {

	public static final String FILEPATH = "kwic_output.txt";

	public void write(List<String> lines) throws IOException {
		// Creates and closes the writer each time to allow for multiple Outputs
		FileWriter writer = new FileWriter(FILEPATH);
		for (String line : lines)
			writer.write(line + '\n');
		writer.close();
	}

}