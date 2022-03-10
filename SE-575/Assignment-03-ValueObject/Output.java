import java.io.FileWriter;
import java.io.IOException;

public class Output implements Observer {

	public static final String FILEPATH = "kwic_output.txt";

	private final FileWriter writer;

	public Output() throws IOException {
		this.writer = new FileWriter(FILEPATH);
	}

	@Override
	public void update(Message msg) {
		if (msg.getBool()) {
			this.closeWriter();
		} else {
			this.write(msg.getString());
		}
	}

	private void write(String line) {
		try {
			if (line.endsWith("\n")) {
				this.writer.write(line);
			} else {
				this.writer.write(line + "\n");
			}
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