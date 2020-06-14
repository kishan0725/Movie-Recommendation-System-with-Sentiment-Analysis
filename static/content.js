function content(e) {
  localStorage.setItem('title', e.getAttribute('title'));
  window.document.location="movie/"+e.getAttribute('title');
}

// function setcontent(){
//     document.getElementById("name").innerHTML = localStorage.getItem('title');
// }