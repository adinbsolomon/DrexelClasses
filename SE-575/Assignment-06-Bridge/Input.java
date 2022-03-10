import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;

public class Input extends Filter {

	public static final String FILEPATH = "kwic.txt";

	private static final File FILE = makeInputFile();

	private Scanner reader;

	public Input(Pipe inPipe, Pipe outPipe) {
		super(inPipe, outPipe);
		this.reader = this.makeInputScanner();
	}

	private static File makeInputFile() {
		File file = new File(FILEPATH);
		try {
			file.createNewFile();
		} catch (IOException ignored) {
		}
		return file;
	}

	@Override
	public void filter() {
		while (this.reader.hasNextLine()) {
			this.outPipe.write(this.reader.nextLine());
		}
		this.outPipe.close();
		this.stop();
	}

	private Scanner makeInputScanner() {
		assert (FILE.exists());
		assert (FILE.canRead());
		Scanner scanner = null;
		try {
			scanner = new Scanner(FILE);
		} catch (FileNotFoundException ignored) {
		}
		return scanner;
	}

}
