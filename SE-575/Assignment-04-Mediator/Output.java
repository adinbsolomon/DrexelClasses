import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class Output {

	public static final String FILEPATH = "kwic_output.txt";

	private FileWriter writer;

	public Output() {
		this.makeWriter();
	}

	public void write(List<String> lines) {
		for (String line : lines) {
			this.writeLine(line);
		}
		this.closeWriter();
	}

	private void makeWriter() {
		try {
			this.writer = new FileWriter(FILEPATH);
		} catch (IOException ignored) {
		}
	}

	private void writeLine(String line) {
		try {
			this.writer.write(line + "\n");
		} catch (IOException ignored) {
		}
	}

	private void closeWriter() {
		try {
			this.writer.close();
		} catch (IOException ignored) {
		}
	}

}