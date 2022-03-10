public abstract class Pipe {

	protected boolean closed;

	public Pipe() {
		this.closed = false;
	}

	public boolean isNotEmptyOrIsNotClosed() {
		return !this.isEmpty() || !this.closed;
	}

	public boolean hasNext() {
		return !this.isEmpty();
	}

	public void close() {
		this.closed = true;
	}

	public abstract void write(String string);

	public abstract String read();

	public abstract boolean isEmpty();

}
