import java.util.Objects;

public class Message {

	public static final boolean DEFAULT_BOOL = false;
	public static final String DEFAULT_STRING = null;
	public static final Message finished = new Message(true);

	private final boolean bool;
	private final String str;

	public Message(boolean bool) {
		this.bool = bool;
		this.str = DEFAULT_STRING;
	}

	public Message(String str) {
		this.str = str;
		this.bool = DEFAULT_BOOL;
	}

	public String getString() {
		if (Objects.equals(this.str, DEFAULT_STRING)) {
			throw new IllegalStateException();
		}
		return this.str;
	}

	public Boolean getBool() {
		return this.bool;
	}

}
