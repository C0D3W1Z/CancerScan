var $messages = $('.messages-content'),
    d, h, m,
    i = 0;

$(window).load(function () {
  $messages.mCustomScrollbar();
  setTimeout(function () {
    fakeMessage();
  }, 100);
});

function insertMessage() {
  msg = $(".message-input").val();
  if ($.trim(msg) == "") {
    return false;
  }
  $('<div class="message message-personal">' + msg + "</div>")
  .appendTo($('.messages'))
  .addClass("new");
  $(".message-input").val(null);
  
  setTimeout(function () {
    fakeMessage();
  }, 1000 + Math.random() * 20 * 100);
  }


$('.message-submit').click(function() {
  insertMessage();
});

$(window).on('keydown', function(e) {
  if (e.which == 13) {
    insertMessage();
    return false;
  }
})

function fakeMessage() {
  if ($('.message-input').val() != '') {
    return false;
  }
  
  var text2 = document.getElementById("textfrompython").textContent;
  console.log(text2)

  var ftext = document.getElementById("textfrompython2").textContent;
  console.log(ftext)
  
  var Fake = [
    "Hello, what's on your mind?",
    text2,
  ]

  if (text2 != "none") {
    Fake[0] = text2; // Replace the first message in the array with the contents of text2
  }
  
  $('<div class="message loading new"><figure class="avatar"><img src="static/assets/img/chatbot.jpeg" /></figure><span></span></div>'
  ).appendTo($('.messages'));
  



  setTimeout(function() {
    $('.message.loading').remove();
    $('<div class="message new"><figure class="avatar"><img src="static/assets/img/chatbot.jpeg" /></figure>' + Fake[i] + '</div>').appendTo($('.messages')).addClass('new');
    
    i++;
  }, 1000 + Math.random() * 20 * 100);

}