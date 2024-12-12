import { v2 as cloudinary } from "cloudinary";

(async function () {
  // Configuration
  cloudinary.config({
    cloud_name: "dys8zymcx",
    api_key: "754948374267897",
    api_secret: "rgvtpvNivIVWoBB7Od2_lE7VzLI", // Click 'View API Keys' above to copy your API secret
  });

  // Upload an image
  const uploadResult = await cloudinary.uploader
    .upload(
      "https://res.cloudinary.com/dys8zymcx/image/upload/v1729272209/Next-file-Uploads/ndf5mrasoluk9xqrhohw.jpg",
      {
        public_id: "Next-file-Uploads/ndf5mrasoluk9xqrhohw",
      }
    )
    .catch((error) => {
      console.log(error);
    });

  console.log(uploadResult);

  // Optimize delivery by resizing and applying auto-format and auto-quality
  const optimizeUrl = cloudinary.url("Next-file-Uploads/ndf5mrasoluk9xqrhohw", {
    fetch_format: "auto",
    quality: "auto",
  });

  console.log(optimizeUrl);

  // Transform the image: auto-crop to square aspect_ratio
  const autoCropUrl = cloudinary.url("Next-file-Uploads/ndf5mrasoluk9xqrhohw", {
    crop: "auto",
    gravity: "auto",
    width: 500,
    height: 500,
  });

  console.log(autoCropUrl);

  const background = cloudinary.image(
    "Next-file-Uploads/ndf5mrasoluk9xqrhohw",
    {
      effect:
        "gen_background_replace:prompt_please set teh background the Makkah sarif",
    }
  );
  console.log(background);
})();

// Next-file-Uploads/ndf5mrasoluk9xqrhohw
