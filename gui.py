from tkinter import *

from chat import get_response
bot = 'Footy'

BG_BLACK = '#ABB2B9'
BG_COLOR = '#345d73'

FONT = 'Calibri 13'
FONT_BOLD = 'Calibri 15 bold'

class ChatbotGUI:

    #Setting up the window of the gui
    def __init__(self):
        self.window = Tk()
        self._setup_homepage()

    def run(self):
        self.window.mainloop()

    def _setup_homepage(self):
        #Setting the title and dimensions of the window
        self.window.title('Footy the Friendly Football Chatbot')
        self.window.configure(width=700, height=700, bg=BG_COLOR)

        #Adding a heading to the window
        header = Label(self.window, bg=BG_COLOR, fg='#f5f6f7', text='Greetings from Footy', font=FONT_BOLD, pady=10)
        header.place(relwidth=1)

        line = Label(self.window, width=480, bg=BG_BLACK)
        line.place(relwidth=1, rely=0.08, relheight=0.01)

        #Creating ta text widget for the user's input and the chatbot's response
        self.chatbox = Text(self.window, width=15, height=2, bg='#f5f6f7', fg='#30414a', font=FONT, padx=5, pady=5)
        self.chatbox.place(relheight=0.7, relwidth=0.95, relx=0.025, rely = 0.1)
        self.chatbox.configure(cursor='arrow', state='disabled')

        #Adding a scrollbar to the text widget
        scrollbar = Scrollbar(self.chatbox)
        scrollbar.place(relheight=1, relx=0.97)
        scrollbar.configure(command=self.chatbox.yview, bg=BG_BLACK)

        #Creating a background for the button and the data entry box
        bottom_area = Label(self.window, bg=BG_BLACK, height=7)
        bottom_area.place(relwidth=0.95, relx=0.025, rely=0.81)

        #Creatinf the data entry box
        self.entry = Entry(bottom_area, bg='#101112', fg='#f7f9fa', font=FONT)
        self.entry.place(relwidth=0.8, relheight=0.95, rely=0.025, relx=0.001)
        self.entry.focus()
        #Enabling the 'Enter' key to have the same functionality as the button
        self.entry.bind('<Return>', self._enter)

        #Creating the button
        enter_button = Button(bottom_area, text='Send', font=FONT_BOLD, bg=BG_COLOR, command=lambda: self._enter(None))
        enter_button.place(relwidth=0.195, relheight=0.95, rely=0.025, relx=0.804)

    #Defining the function for retrieving the data in the textbox
    def _enter(self, event):
        data = self.entry.get()
        self._provide_data(data, 'You')

    #Defining the function for printing data in the text widget
    def _provide_data(self, data, sender):
        if not data:
            return

        self.entry.delete(0, END)
        data1 = f'{sender}: {data}\n\n'
        self.chatbox.configure(state='normal')
        self.chatbox.insert(END, data1)
        self.chatbox.configure(state='disabled')

        data2 = f'{bot}: {get_response(data)}\n\n'
        self.chatbox.configure(state='normal')
        self.chatbox.insert(END, data2)
        self.chatbox.configure(state='disabled')

        self.chatbox.see(END)

if __name__ == '__main__':
    chatbotgui = ChatbotGUI()
    chatbotgui.run()