import java.io.FileWriter;
import java.io.IOException;

public class Output extends Filter {

	public static final String FILEPATH = "kwic_output.txt";

	private FileWriter writer;

	public Output(Pipe inPipe, Pipe outPipe) {
		super(inPipe, outPipe);
		this.makeWriter();
	}

	@Override
	public void filter() {
		while (this.inPipe.isNotEmptyOrIsNotClosed()) {
			if (this.inPipe.hasNext()) {
				this.writeLine(this.inPipe.read());
			}
		}
		this.closeWriter();
		this.stop();
	}

	private void makeWriter() {
		try {
			this.writer = new FileWriter(FILEPATH);
		} catch (IOException ignored) {
		}
	}

	private void writeLine(String line) {
		try {
			this.writer.write(line + '\n');
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