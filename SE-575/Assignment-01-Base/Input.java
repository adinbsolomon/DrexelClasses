import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class Input {

	public static final String FILEPATH = "kwic.txt";

	private static final File FILE = makeInputFile();

	private final Scanner reader;

	public Input() {
		this.reader = this.makeInputScanner();
	}

	public ArrayList<String> read() {
		ArrayList<String> lines = new ArrayList<String>();
		while (this.reader.hasNextLine())
			lines.add(this.reader.nextLine());
		return lines;
	}

	private Scanner makeInputScanner() {
		assert (FILE.exists());
		assert (FILE.canRead());
		Scanner scanner = null;
		try {
			scanner = new Scanner(FILE);
		} catch (FileNotFoundException e) {
			// Asserting that the file exists and can be read should be sufficient
			// - this just removes the 'throws exception' from the declaration
		}
		return scanner;
	}

	private static File makeInputFile() {
		File file = new File(FILEPATH);
		try {
			file.createNewFile();
		} catch (IOException e) {
		}
		return file;
	}

}
