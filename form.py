from tkinter import *
from LSTM import LR_Model

form = Tk()
form.title("Nhập tham số:")
form.geometry("1000x500")

lable_file_train = Label(form, text = "Tập dữ liệu huấn luyện:")
lable_file_train.grid(row = 1, column = 1, padx = 40, pady = 10, sticky=W)
textbox_file_train = Entry(form)
textbox_file_train.grid(row = 1, column = 2)

lable_file_test = Label(form, text = "Tập dữ liệu kiểm tra:")
lable_file_test.grid(row = 2, column = 1, padx = 40, pady = 10, sticky=W)
textbox_file_test = Entry(form)
textbox_file_test.grid(row = 2, column = 2)

lable_know_attributes = Label(form, text = "Thuộc tính có thông tin:")
lable_know_attributes.grid(row = 3, column = 1, padx = 40, pady = 10, sticky=W )
textbox_know_attributes = Entry(form)
textbox_know_attributes.grid(row = 3, column = 2)

lable_unknow_attributes = Label(form, text = "Thuộc tính chưa có thông tin:")
lable_unknow_attributes.grid(row = 4, column = 1, padx = 40, pady = 10, sticky=W)
textbox_unknow_attributes = Entry(form)
textbox_unknow_attributes.grid(row = 4, column = 2)

lable_pred_attributes = Label(form, text = "Thuộc tính dự đoán:")
lable_pred_attributes.grid(row = 5, column = 1, padx = 40, pady = 10, sticky=W)
textbox_pred_attributes = Entry(form)
textbox_pred_attributes.grid(row = 5, column = 2)

lable_folder_name = Label(form, text = "Thư mục xuất kết quả:")
lable_folder_name.grid(row = 6, column = 1, padx = 40, pady = 10, sticky=W)
textbox_folder_name = Entry(form)
textbox_folder_name.grid(row = 6, column = 2)

lable_numdays = Label(form, text = "Số ngày đã biết:")
lable_numdays.grid(row = 7, column = 1, padx = 40, pady = 10, sticky=W)
textbox_numdays = Entry(form)
textbox_numdays.grid(row = 7, column = 2)

lable_afterdays = Label(form, text = "Dự đoán sau bao nhiêu ngày:")
lable_afterdays.grid(row = 8, column = 1,padx = 40, pady = 10, sticky=W)
textbox_afterdays = Entry(form)
textbox_afterdays.grid(row = 8, column = 2)

def split_string(s):
    str = s.split(",")
    result = []
    for i in range(len(str)):
        result.append(str[i].strip())
    return result

def getform():
    file_train = textbox_file_train.get()
    file_train = file_train.strip()

    file_test = textbox_file_test.get()
    file_test = file_test.strip()

    text_know_attributes = textbox_know_attributes.get()
    know_attributes = split_string(text_know_attributes)

    text_unknow_attributes = textbox_unknow_attributes.get()
    unknow_attributes = split_string(text_unknow_attributes)

    pred_attribute = textbox_pred_attributes.get()
    pred_attribute = split_string(pred_attribute)

    foldername = textbox_folder_name.get()
    foldername = foldername.strip()

    max_numdays = int(textbox_numdays.get())
    max_afterdays = int(textbox_afterdays.get())

    LR_Model(file_train, file_test, know_attributes, unknow_attributes, pred_attribute, foldername, max_numdays,
             max_afterdays, nomalize=False)

button_submit = Button(form, text = 'Submit', command = getform)
button_submit.grid(row = 9, column = 1, pady = 20)

form.mainloop()